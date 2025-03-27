import os
import requests
import shutil

def download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {local_path}")

# Base URL for raw GitHub content
base_url = "https://raw.githubusercontent.com/IDEA-Research/DWPose/onnx/ControlNet-v1-1-nightly"

# Files to download
files = [
    "/annotator/dwpose/__init__.py",
    "/annotator/dwpose/wholebody.py",
    "/annotator/dwpose/onnxdet.py",
    "/annotator/dwpose/onnxpose.py",
    "/annotator/__init__.py",
]

# Download each file
root_dir = os.path.dirname(os.path.abspath(__file__))
for file_path in files:
    url = base_url + file_path
    local_path = os.path.join(root_dir, "dwpose", file_path.lstrip('/'))
    download_file(url, local_path)

# Restore checkpoints
if os.path.exists('temp_ckpts'):
    os.makedirs('dwpose/annotator/ckpts', exist_ok=True)
    for file in os.listdir('temp_ckpts'):
        src = os.path.join('temp_ckpts', file)
        dst = os.path.join('dwpose/annotator/ckpts', file)
        shutil.copy2(src, dst)
    shutil.rmtree('temp_ckpts')
    print("Restored checkpoint files")
