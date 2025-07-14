import argparse
import os

from huggingface_hub import hf_hub_download, login


def download_file(repo_id, filename, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False,
        )
        # Move the file to the correct location if needed
        if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
            os.replace(downloaded_path, local_path)

        # Verify file size
        actual_size = os.path.getsize(local_path)
        print(f"Downloaded {local_path}, size: {actual_size} bytes")
        if actual_size < 1000000:  # Less than 1MB
            print("Warning: File size seems too small for a model file")
            return False
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e!s}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download DWPose model files")
    parser.add_argument("--token", required=True, help="Hugging Face token")
    args = parser.parse_args()

    # Login to Hugging Face
    login(token=args.token)

    # Model information
    models = {
        "yolox_l.onnx": {"repo_id": "yzd-v/DWPose", "filename": "yolox_l.onnx"},
        "dw-ll_ucoco_384.onnx": {
            "repo_id": "yzd-v/DWPose",
            "filename": "dw-ll_ucoco_384.onnx",
        },
    }

    # Download each model
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ckpts_dir = os.path.join(root_dir, "DWPose", "annotator", "ckpts")

    for model_name, info in models.items():
        local_path = os.path.join(ckpts_dir, model_name)
        print(f"Downloading {model_name}...")
        if not download_file(info["repo_id"], info["filename"], local_path):
            print(f"Failed to download {model_name}")
            exit(1)


if __name__ == "__main__":
    main()
