#!/usr/bin/env python3
"""
Upload WebDataset to HuggingFace datasets.
"""

import os
import json
import argparse
import shutil
import tempfile
from huggingface_hub import HfApi, upload_folder
import webdataset as wds
from tqdm import tqdm


def create_dataset_card(
    webdataset_path, output_path, dataset_name, language_codes=None
):
    """
    Create a dataset card for HuggingFace.

    Args:
        webdataset_path: Path to WebDataset shards
        output_path: Path to save dataset card
        dataset_name: Name of the dataset
        language_codes: List of language codes in the dataset
    """
    # If path is a directory, find all .tar files
    if os.path.isdir(webdataset_path):
        shards = [
            os.path.join(webdataset_path, f)
            for f in os.listdir(webdataset_path)
            if f.endswith(".tar")
        ]
        shards.sort()
    else:
        shards = [webdataset_path]

    if not shards:
        raise ValueError(f"No WebDataset shards found at {webdataset_path}")

    # Get sample count
    sample_count = 0
    languages = set()
    projects = set()

    # Examine first shard to get metadata
    print(f"Reading metadata from shard: {shards[0]}")

    try:
        # Use a direct file path approach instead of URL-like path
        dataset = wds.WebDataset(shards[0], shardshuffle=False).decode()

        for sample in dataset:
            sample_count += 1

            # Parse metadata if available
            if "json" in sample:
                try:
                    metadata = json.loads(sample["json"])
                    if "language_code" in metadata:
                        languages.add(metadata["language_code"])
                    if "project_name" in metadata:
                        projects.add(metadata["project_name"])
                except:
                    pass
    except Exception as e:
        print(f"Warning: Error reading shard metadata: {e}")
        print("Attempting to read metadata from manifest.json instead...")

        # Try to get metadata from manifest.json
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest_path = os.path.join(project_root, "output", "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                for segment in manifest.get("segments", []):
                    metadata = segment.get("segment_metadata", {})
                    if "language_code" in metadata:
                        languages.add(metadata["language_code"])
                    if "project_name" in metadata:
                        projects.add(metadata["project_name"])

                # Estimate sample count based on manifest
                sample_count = len(manifest.get("segments", []))
                print(f"Found {sample_count} samples in manifest.json")
            except Exception as e:
                print(f"Warning: Error reading manifest.json: {e}")

    # Add language codes if provided
    if language_codes:
        languages.update(language_codes)

    # Create dataset card
    card = f"""---
language:
{chr(10).join(f"- {lang}" for lang in sorted(languages))}
license: cc-by-sa-4.0
datasets:
- {dataset_name}
---

# {dataset_name}

**This dataset is still being generated and currently includes only test files**

This dataset contains sign language videos from the Digital Bible Library (DBL), processed for machine learning applications. The dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

## Dataset Description

- **Size**: {len(shards)} shards, approximately {sample_count * len(shards)} samples
- **Languages**: {", ".join(sorted(languages))}
- **Projects**: {", ".join(sorted(projects))}
- **License**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Dataset Structure

Each sample in the dataset contains:
- `.original.mp4`: Original video segment
- `.pose.mp4`: Video with pose keypoints visualization
- `.mask.mp4`: Video with segmentation masks
- `.segmentation.mp4`: Video with combined segmentation
- `.json`: Metadata including copyright (from rights_holder), license, and source (from url) information

## Usage

```python
from datasets import load_dataset
import json

# Load dataset from Hugging Face Hub
dataset = load_dataset("bible-nlp/sign-bibles")

# Iterate through samples
for sample in dataset["train"]:
    # Access components
    original_video = sample["original.mp4"]
    pose_video = sample["pose.mp4"]
    mask_video = sample["mask.mp4"]
    segmentation_video = sample["segmentation.mp4"]
    metadata = sample["segment_metadata"]
```

## License and Attribution

This dataset is derived from the Digital Bible Library (DBL) sign language content.
Each sample includes copyright and license information in its metadata.

The dataset is provided under a CC BY-SA 4.0 license, which allows for:
- Sharing: Copy and redistribute the material in any medium or format
- Adaptation: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- ShareAlike: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original

Please refer to the individual sample metadata for specific attribution requirements.
"""

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)

    return output_path


def upload_to_huggingface(webdataset_path, dataset_name, token=None):
    """
    Upload WebDataset shards to HuggingFace.

    Args:
        webdataset_path: Path to WebDataset shards
        dataset_name: Name of the dataset on HuggingFace
        token: HuggingFace API token
    """
    print(f"Uploading WebDataset from {webdataset_path} to {dataset_name}")

    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy shards to temporary directory
        if os.path.isdir(webdataset_path):
            # Get absolute path to avoid path issues
            webdataset_path = os.path.abspath(webdataset_path)
            shards = [
                os.path.join(webdataset_path, f)
                for f in os.listdir(webdataset_path)
                if f.endswith(".tar")
            ]
            shards.sort()
        else:
            # Get absolute path to avoid path issues
            shards = [os.path.abspath(webdataset_path)]

        if not shards:
            raise ValueError(f"No WebDataset shards found at {webdataset_path}")

        print(f"Copying {len(shards)} shards to temporary directory...")
        for i, shard in enumerate(tqdm(shards)):
            print(f"Copying shard {i + 1}/{len(shards)}: {shard}")
            shutil.copy(shard, os.path.join(tmp_dir, f"shard_{i:05d}.tar"))

        # Copy manifest.json if it exists
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest_path = os.path.join(project_root, "output", "manifest.json")
        if os.path.exists(manifest_path):
            print("Copying manifest.json to upload directory...")
            shutil.copy(manifest_path, os.path.join(tmp_dir, "manifest.json"))
        else:
            print(f"Warning: manifest.json not found at {manifest_path}")

        # Create dataset card
        readme_path = os.path.join(tmp_dir, "README.md")
        create_dataset_card(webdataset_path, readme_path, dataset_name)

        # Upload to HuggingFace
        print(f"Uploading to HuggingFace dataset: {dataset_name}")
        api = HfApi(token=token)

        try:
            # Upload files
            upload_folder(
                folder_path=tmp_dir,
                repo_id=dataset_name,
                repo_type="dataset",
                token=token,
                ignore_patterns=["*.pyc", "__pycache__", ".git*"],
                commit_message="Upload dataset files",
            )
            print(f"Successfully uploaded to HuggingFace: {dataset_name}")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check if your token has write access to the repository")
            print(
                "2. Verify that you have the correct organization name in the dataset_name"
            )
            print(
                "3. Try logging in with 'huggingface-cli login' before running this script"
            )
            print("4. Check your internet connection")
            raise


def main():
    """Main function to upload WebDataset to HuggingFace."""
    parser = argparse.ArgumentParser(description="Upload WebDataset to HuggingFace")
    parser.add_argument("webdataset_path", help="Path to WebDataset shards")
    parser.add_argument(
        "dataset_name",
        help="Name of the dataset on HuggingFace (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HUGGINGFACE_TOKEN environment variable)",
    )
    args = parser.parse_args()

    # Get token from environment variable if not provided
    token = args.token
    if not token:
        # Try system environment variable
        token = os.environ.get("HUGGINGFACE_TOKEN")

        # Try user environment variable if system one is not available
        if not token:
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "powershell",
                        "-Command",
                        "[Environment]::GetEnvironmentVariable('HUGGINGFACE_TOKEN', 'User')",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    token = result.stdout.strip()
            except Exception as e:
                print(f"Warning: Error accessing user environment variables: {e}")

    if not token:
        print(
            "Warning: No HuggingFace API token provided. You may need to login interactively."
        )

    # Upload to HuggingFace
    upload_to_huggingface(args.webdataset_path, args.dataset_name, token)


if __name__ == "__main__":
    main()
