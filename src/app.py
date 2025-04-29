import sys
import os
import re
from datetime import datetime
import shutil
import requests  # Add requests library for API calls

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import asyncio
from utils.taobao_crawler import TaobaoCrawler
from utils.data_processor import DataProcessor
from agents.fashion_agent import FashionAgent
import pandas as pd
import json
from PIL import Image
import io
import base64
import numpy as np
import torch

# Add OpenAI API support
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Add imports needed for agent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Global variables
crawler = None
data_processor = None
fashion_agent = None
clothing_data = None
current_mode = "taobao"  # Default recommendation mode

MODEL_OPTIONS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "AI21-Jamba-1.5-Large",
    "AI21-Jamba-1.5-Mini",
    "Codestral-2501",
    "Cohere-command-r",
    "Ministral-3B",
    "Mistral-Large-2411",
    "Mistral-Nemo",
    "Mistral-small"
]

# Initialize OpenAI client
openai_client = None
openai_model = "gpt-4.1-mini"  # Default model

def initialize_openai_client():
    """Initialize OpenAI client"""
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not found")
        return False
    
    openai_client = OpenAI(api_key=api_key)
    return True

def set_openai_model(model_name):
    """Set OpenAI model"""
    global openai_model
    openai_model = model_name
    print(f"OpenAI model set to: {openai_model}")

async def start_crawler():
    """Start crawler and return login page URL"""
    global crawler
    try:
        crawler = TaobaoCrawler()
        # Direct login
        if crawler.login():
            return "Login successful!"
        return "Login failed, please try again."
    except Exception as e:
        print(f"Error in start_crawler: {str(e)}")
        return f"Error occurred: {str(e)}"

async def check_login():
    """Check login status"""
    if crawler:
        return "Logged in"
    return "Not logged in"

async def process_data():
    """Process crawled data"""
    global data_processor, clothing_data
    if not crawler:
        return []
        
    # Initialize data processor
    data_processor = DataProcessor(data_dir="data")
    
    # Get data
    items = crawler.get_purchase_history(days=30)
    if items:
        try:
            # Convert items to DataFrame
            items_df = pd.DataFrame(items)
            # Process data
            clothing_data = data_processor.process_data(items_df)
            # Close browser
            crawler.close()
            # Return image URL list
            return get_image_urls()
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return []
    return []

def get_image_urls():
    """Get all image URLs"""
    if clothing_data is not None:
        return list(clothing_data['image_url'].values)
    return []

def update_model(selected_model):
    # Set OpenAI model (for image analysis)
    if selected_model in ["gpt-4.1-mini", "gpt-4.1-turbo", "gpt-4.0-turbo", "gpt-4o", "gpt-4o-mini"]:
        set_openai_model(selected_model)
    
    # Also keep the original GitHub model setting (for other functions)
    os.environ["GITHUB_MODEL"] = selected_model
    return f"Current selected model: {selected_model}"

def upload_to_imgbb(image_path):
    """Upload image to ImgBB and get public URL"""
    try:
        # Use user-provided API Key
        imgbb_key = os.getenv("IMGBB_KEY", "1fd2bb0a1db93995e5736b5e275f7f28")  # User-provided API Key
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
            
        # Convert image to base64 encoding
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Upload to ImgBB
        res = requests.post(
            'https://api.imgbb.com/1/upload',
            data={
                'key': imgbb_key,
                'image': base64_image,
            }
        )
        
        # Print response for debugging
        print(f"ImgBB API Response: {res.text}")
        
        if res.status_code == 200:
            response_json = res.json()
            if response_json.get('success'):
                # Return only the image URL
                return response_json['data']['url']
            else:
                return f"ImgBB upload failed: {response_json.get('error', {}).get('message', 'Unknown error')}"
        else:
            return f"ImgBB upload failed, status code: {res.status_code}"
    except Exception as e:
        print(f"ImgBB upload error: {str(e)}")
        return f"Error during ImgBB upload process: {str(e)}"

async def get_recommendation(style_preference: str, temperature: float = None, mood: str = None, tab: str = "taobao"):
    """Get clothing recommendations
    
    Args:
        style_preference: Style preference
        temperature: Temperature
        mood: Mood
        tab: Tab type, 'taobao' uses CSV data, 'physical' uses TXT data
    """
    global clothing_data
    result_images = []
    
    try:
        # Initialize recommendation agent
        if tab == "taobao":
            # In taobao mode, use global clothing_data (DataFrame)
            if clothing_data is None:
                return [], "Please process Taobao data first"
            fashion_agent = FashionAgent(clothing_data)
            print(f"Using Taobao CSV data with {len(clothing_data)} records")
        elif tab == "physical":
            # In physical mode, read image analysis text result
            txt_path = "data/upload_process_txt.txt"
            
            # Check if file exists
            if not os.path.exists(txt_path):
                return [], f"File does not exist: {txt_path}, please upload and analyze an image first"
                
            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    text_description = file.read()
                print(f"Successfully read image analysis text, content length: {len(text_description)} characters")
                if len(text_description.strip()) == 0:
                    return [], "Image analysis text is empty, please upload and analyze an image again"
                    
                fashion_agent = FashionAgent(text_description)
            except Exception as e:
                return [], f"Failed to read image analysis text: {str(e)}"
        else:
            return [], f"Invalid tab type: {tab}. Please use 'taobao' or 'physical'."
        
        # Get recommendation results
        result = await fashion_agent.process_request(
            style_preference=style_preference,
            temperature=temperature,
            mood=mood
        )
        
        # Parse image URLs from the recommendation result
        recommended_images = re.findall(r'https?://[^\s]+\.jpg', result)
        
        # Replace image URLs with placeholders when printing
        print_result = re.sub(r'https?://[^\s]+\.jpg', '[IMAGE_URL]', result)
        print("LLM result:", print_result[:100] + "..." if len(print_result) > 100 else print_result)
        
        return recommended_images, result
        
    except Exception as e:
        print(f"Error during recommendation process: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error getting recommendation: {str(e)}"

async def analyze_image_with_openai(image_url):
    """Use OpenAI API to analyze image and generate detailed description"""
    global openai_client, openai_model
    
    try:
        # Ensure OpenAI client is initialized
        if openai_client is None:
            if not initialize_openai_client():
                return "Error: OpenAI API key not set, please set OPENAI_API_KEY in environment variables"
        
        print(f"Using OpenAI API ({openai_model}) to analyze image: {image_url}")
        
        # Create API request
        response = openai_client.responses.create(
            model=openai_model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": image_url
                    },
                ],
            }],
        )
        
        # Extract analysis result
        analysis_result = response.output_text
        print(f"OpenAI analysis complete, result length: {len(analysis_result)}")
        
        
        return analysis_result
    except Exception as e:
        print(f"OpenAI image analysis error: {str(e)}")
        return f"Error during image analysis process: {str(e)}"

async def process_uploaded_images(image_path):
    """Process uploaded images and analyze clothing features"""
    try:
        if not image_path:
            return "Please upload an image"
            
        print(f"Received image path: {image_path}")
        
        # Use ImgBB to upload image
        imgbb_url = upload_to_imgbb(image_path)
        
        # Check if ImgBB upload was successful - if the returned URL is a string and doesn't start with error message
        if isinstance(imgbb_url, str) and not imgbb_url.startswith("ImgBB upload failed") and not imgbb_url.startswith("Error during ImgBB upload process"):
            # Upload successful, use OpenAI API for image analysis
            print("Image upload successful, starting OpenAI analysis...")
            result_message = await analyze_image_with_openai(imgbb_url)
            
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Save analysis result to file for get_recommendation function to use
            try:
                with open("data/upload_process_txt.txt", "w", encoding="utf-8") as f:
                    f.write(result_message)
                print(f"Analysis result saved to file, length: {len(result_message)} characters")
            except Exception as e:
                print(f"Failed to save analysis result to file: {str(e)}")
        else:
            result_message = f"ImgBB upload failed: {imgbb_url}"
        
        return result_message
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}"

# Non-async wrapper function for direct calling of async functions
def sync_get_recommendation(style, temp, mood):
    """Synchronous wrapper function for Gradio button click event"""
    global current_mode
    print(f"Using {current_mode} mode for clothing recommendation")
    
    try:
        # Use asyncio.new_event_loop().run_until_complete to run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        images, text = loop.run_until_complete(get_recommendation(style, temp, mood, tab=current_mode))
        loop.close()
        
        return images, text
    except Exception as e:
        print(f"Recommendation process error: {str(e)}")
        return [], f"Error during recommendation process: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Fashion Recommendation System")
    
    with gr.Row():
        # Left Column
        with gr.Column(scale=1):
            # Using Tabs to organize different data acquisition methods
            with gr.Tabs() as tabs:
                # Tab 1: Online Wardrobe (Taobao Data)
                with gr.Tab("Online Wardrobe (Taobao)") as taobao_tab:
                    gr.Markdown("## Get Taobao Purchase History")
                    start_button = gr.Button("Login to Taobao")
                    login_status = gr.Textbox(label="Login Status", interactive=False)
                    process_button = gr.Button("Get Clothing Data")
                    
                    # Display Taobao clothing images
                    gr.Markdown("## My Taobao Clothing")
                    taobao_gallery = gr.Gallery(label="Taobao Clothing", show_label=False)
                
                # Tab 2: Offline Wardrobe (Image Upload)
                with gr.Tab("Offline Wardrobe (Physical Clothes)") as physical_tab:
                    gr.Markdown("## Upload Clothing Image")
                    image_upload = gr.Image(
                        label="Upload Clothing Image",
                        type="filepath"  # Return local file path
                    )
                    upload_button = gr.Button("Analyze Image")
                    
                    # Display analysis results
                    gr.Markdown("## Clothing Analysis Results")
                    analysis_text = gr.Textbox(
                        label="Clothing Analysis",
                        interactive=False,
                        lines=10,
                        placeholder="Waiting for analysis results..."
                    )
            
            # Display current recommendation mode
            current_mode_text = gr.Markdown("**Current Mode: Taobao**")
            
            # Recommendation Settings (shared by both tabs)
            gr.Markdown("## Recommendation Settings")
            style_input = gr.Textbox(label="Desired Style")
            temperature_input = gr.Number(label="Current Temperature (Optional)")
            mood_input = gr.Textbox(label="Current Mood (Optional)")
            
            model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=os.getenv('API_HOST', 'github'),
                label="Select Model"
            )
            recommend_button = gr.Button("Get Recommendations")
        
        # Right Column
        with gr.Column(scale=1):
            # Recommendation Results
            gr.Markdown("## Recommended Outfits")
            recommendation_gallery = gr.Gallery(label="Recommended Outfits", show_label=False)
            recommendation_text = gr.Textbox(label="Recommendation Explanation", interactive=False, lines=10)
    
    # Bind events
    start_button.click(
        fn=start_crawler,
        outputs=login_status
    )
    
    process_button.click(
        fn=process_data,
        outputs=taobao_gallery
    )
    
    upload_button.click(
        fn=process_uploaded_images,
        inputs=image_upload,
        outputs=analysis_text
    )
    
    # Use the selected mode for recommendations
    recommend_button.click(
        fn=sync_get_recommendation,
        inputs=[style_input, temperature_input, mood_input],
        outputs=[recommendation_gallery, recommendation_text]
    )
    
    model_dropdown.change(
        fn=update_model,
        inputs=model_dropdown,
        outputs=[]
    )
    
    # Define function to update mode
    def update_mode(mode):
        global current_mode
        current_mode = mode
        return f"**Current Mode: {mode.capitalize()}**"
    
    # Automatically update recommendation mode when tabs are switched
    taobao_tab.select(
        fn=lambda: update_mode("taobao"),
        inputs=None,
        outputs=current_mode_text
    )
    
    physical_tab.select(
        fn=lambda: update_mode("physical"),
        inputs=None,
        outputs=current_mode_text
    )

# run on hugging face spaces and local machine 
if __name__ == "__main__":
    print(f"Using API_HOST: {os.getenv('API_HOST', 'github')}")
    print(f"Using GITHUB_MODEL: {os.getenv('GITHUB_MODEL', 'gpt-4o')}")

    # Detect running environment
    if os.getenv('SPACE_ID'):
        # Hugging Face Spaces environment
        demo.launch()
    else:
        # Local environment
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:2802"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:2802"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"

        # Local run configuration
        is_share = False  # Can be modified as needed
        server_name = "0.0.0.0" if is_share else "127.0.0.1"
        
        # Add a static file service to make uploads directory accessible via web
        demo.queue().launch(
            server_name=server_name,
            share=is_share,
            show_error=True,
            favicon_path=None,
            # Set static file directory
            root_path="/"
        )