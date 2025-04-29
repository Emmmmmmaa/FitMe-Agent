'''can use huggingface to call the model instead of downloading the model'''


# import os
# import requests
# from tqdm import tqdm

# def download_file(url, filename):
#     """Download file with progress bar"""
#     response = requests.get(url, stream=True)
#     total_size = int(response.headers.get('content-length', 0))
    
#     with open(filename, 'wb') as file, tqdm(
#         desc=filename,
#         total=total_size,
#         unit='iB',
#         unit_scale=True,
#         unit_divisor=1024,
#     ) as bar:
#         for data in response.iter_content(chunk_size=1024):
#             size = file.write(data)
#             bar.update(size)

# def main():
#     # Create models directory if it doesn't exist
#     os.makedirs('models', exist_ok=True)
    
#     # SAM model URL
#     sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
#     sam_path = os.path.join('models', 'sam_vit_h_4b8939.pth')
    
#     # Download SAM model if it doesn't exist
#     if not os.path.exists(sam_path):
#         print("Downloading SAM model...")
#         download_file(sam_url, sam_path)
#         print("SAM model downloaded successfully!")
#     else:
#         print("SAM model already exists!")

# if __name__ == "__main__":
#     main() 