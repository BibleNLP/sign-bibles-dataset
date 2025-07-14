import requests
import shutil
from pathlib import Path


def download_file(url: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url}")
    local_path.write_bytes(response.content)
    print(f"Downloaded {local_path}")


# Base URL for raw GitHub files
base_url = "https://raw.githubusercontent.com/IDEA-Research/DWPose/onnx"

# Files to download from GitHub
files = [
    "ControlNet-v1-1-nightly/annotator/dwpose/__init__.py",
    "ControlNet-v1-1-nightly/annotator/dwpose/wholebody.py",
    "ControlNet-v1-1-nightly/annotator/dwpose/onnxdet.py",
    "ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py",
    "ControlNet-v1-1-nightly/annotator/dwpose/util.py",
]

# Determine project root
root_dir = Path(__file__).resolve().parent
dwpose_root = root_dir / "DWPose"

# Download each file to the proper local path under DWPose/
for file_path in files:
    url = f"{base_url}/{file_path}"
    relative_subpath = Path(
        *file_path.split("/")[2:]
    )  # skip ControlNet-v1-1-nightly/annotator
    local_path = dwpose_root / "annotator" / relative_subpath
    download_file(url, local_path)

# Restore checkpoints if previously saved in temp_ckpts/
temp_ckpts = root_dir / "temp_ckpts"
ckpt_dir = dwpose_root / "annotator" / "ckpts"
if temp_ckpts.exists():
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ckpt_file in temp_ckpts.iterdir():
        shutil.copy2(ckpt_file, ckpt_dir / ckpt_file.name)
    shutil.rmtree(temp_ckpts)
    print("Restored checkpoint files")

# Ensure __init__.py files exist for proper Python imports
for pkg_init in [
    dwpose_root / "annotator" / "__init__.py",
    dwpose_root / "annotator" / "dwpose" / "__init__.py",
]:
    pkg_init.parent.mkdir(parents=True, exist_ok=True)
    pkg_init.touch(exist_ok=True)
    print(f"Ensured {pkg_init}")
