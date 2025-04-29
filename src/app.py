import sys
import os
import re

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import asyncio
from utils.taobao_crawler import TaobaoCrawler
from utils.data_processor import DataProcessor
from agents.fashion_agent import FashionAgent
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv(override=True)



# 全局变量
crawler = None
data_processor = None
fashion_agent = None
clothing_data = None

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
    os.environ["GITHUB_MODEL"] = selected_model
    return f"Current selected model: {selected_model}"

async def get_recommendation(style_preference: str, temperature: float = None, mood: str = None):
    """Get clothing recommendations"""
    global fashion_agent
    if clothing_data is not None:
        try:
            # Initialize recommendation agent
            fashion_agent = FashionAgent(clothing_data)
            # Get recommendations
            result = await fashion_agent.process_request(
                style_preference=style_preference,
                temperature=temperature,
                mood=mood
            )
            print("LLM result:",result)
            # Parse recommendation results
            recommended_images = re.findall(r'https?://[^\s]+\.jpg', result)
            print("Extracted image URLs:", recommended_images)
            return recommended_images, result
            
        except Exception as e:
            print(f"Error in get_recommendation: {str(e)}")
            return [], f"Error getting recommendations: {str(e)}"
    return [], "Please process data first"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Fashion Recommendation System")
    
    with gr.Row():
        with gr.Column():
            # Crawler section
            gr.Markdown("## 1. Get Taobao Data")
            start_button = gr.Button("Start Login")
            login_status = gr.Textbox(label="Login Status", interactive=False)
            process_button = gr.Button("Get Clothing Data")
            
            # Display all clothing images
            gr.Markdown("## 2. My Clothing")
            gallery = gr.Gallery(label="My Clothing", show_label=False)
            
            # Recommendation section
            gr.Markdown("## 3. Get Recommendations")
            style_input = gr.Textbox(label="Desired Style")
            temperature_input = gr.Number(label="Current Temperature (Optional)")
            mood_input = gr.Textbox(label="Current Mood (Optional)")
            model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=os.getenv('API_HOST', 'github'),
                label="Select Model"
            )
            recommend_button = gr.Button("Get Recommendations")
            
            # Display recommendation results
            gr.Markdown("## 4. Recommended Outfits")
            recommendation_gallery = gr.Gallery(label="Recommended Outfits", show_label=False)
            recommendation_text = gr.Textbox(label="Recommendation Explanation", interactive=False, lines=10)
    
    # Bind events
    start_button.click(    
        start_crawler,
        outputs=login_status
    )
    
    process_button.click(
        process_data,
        outputs=gallery
    )
    
    model_dropdown.change(
        update_model,
        inputs=model_dropdown,
    )
    recommend_button.click(
        get_recommendation,
        inputs=[style_input, temperature_input, mood_input],
        outputs=[recommendation_gallery, recommendation_text]
    )

# run on hugging face spaces and local machine 
if __name__ == "__main__":
    print(f"Using API_HOST: {os.getenv('API_HOST', 'github')}")
    print(f"Using GITHUB_MODEL: {os.getenv('GITHUB_MODEL', 'gpt-4o')}")

    # 检测运行环境
    if os.getenv('SPACE_ID'):
        # Hugging Face Spaces 环境
        demo.launch()
    else:
        # 本地环境
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:2802"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:2802"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"

        # 本地运行配置
        is_share = True  # 可以根据需要修改
        server_name = "0.0.0.0" if is_share else "127.0.0.1"
        
        demo.queue().launch(
            server_name=server_name,
            share=is_share,
            show_error=True
        )