import os
import torch
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file from a URL to a local file."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download SAM-HQ model
    sam_url = "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    sam_path = os.path.join("models", "sam_hq_vit_h.pth")
    if not os.path.exists(sam_path):
        print("Downloading SAM-HQ model...")
        download_file(sam_url, sam_path)
    
    # Create GroundingDINO config directory structure
    config_dir = os.path.join("GroundingDINO", "groundingdino", "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Download GroundingDINO configuration files
    config_files = {
        "GroundingDINO_SwinT_OGC.py": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "__init__.py": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/__init__.py"
    }
    
    for filename, url in config_files.items():
        filepath = os.path.join(config_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            download_file(url, filepath)
            
    # Create empty __init__.py files for Python package structure
    init_paths = [
        os.path.join("GroundingDINO", "__init__.py"),
        os.path.join("GroundingDINO", "groundingdino", "__init__.py"),
    ]
    
    for init_path in init_paths:
        if not os.path.exists(init_path):
            os.makedirs(os.path.dirname(init_path), exist_ok=True)
            with open(init_path, 'w') as f:
                pass  # Create empty file

if __name__ == "__main__":
    main()
