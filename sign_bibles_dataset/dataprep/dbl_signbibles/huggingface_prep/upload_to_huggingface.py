#!/usr/bin/env python3
"""Upload WebDataset to HuggingFace datasets."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import HTTPError
from tqdm import tqdm

import webdataset as wds
from sign_bibles_dataset.dataprep.dbl_signbibles.huggingface_prep.build_dataset_card import (
    build_dataset_card_markdown_and_yaml,
)


def create_dataset_card(
    webdataset_path: str,
    output_path: str,
    dataset_name: str,
    language_codes: list[str] | None = None,
) -> str:
    """Create and write dataset card file based on ALL shard metadata."""
    shards = find_shards_recursive(webdataset_path)
    total_samples = 0
    languages = set()
    projects = set()

    print(f"Found {len(shards)} shards. Parsing all for metadata...")

    for shard in tqdm(shards, desc="Parsing shard metadata"):
        sample_count, shard_languages, shard_projects = parse_shard_metadata(shard)
        total_samples += sample_count
        languages.update(shard_languages)
        projects.update(shard_projects)

    if language_codes:
        languages.update(language_codes)

    card_text = build_dataset_card_markdown_and_yaml(
        dataset_name=dataset_name,
        shard_count=len(shards),
        total_sample_count=total_samples,
        languages=languages,
        projects=projects,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_text)

    print(f"✅ Dataset card written to {output_path}")
    return output_path


def find_shards_recursive(root_path: str | Path) -> list[str]:
    """Recursively find all .tar shards in the directory."""
    root = Path(root_path)
    shards = sorted(str(p) for p in root.rglob("*.tar"))
    if not shards:
        raise ValueError(f"No WebDataset shards found at {root}")
    return shards


def parse_shard_metadata(shard_path: str) -> tuple[int, set[str], set[str]]:
    """Parse first shard to count samples, languages, and project names."""
    sample_count = 0
    languages = set()
    projects = set()

    # print(f"Reading metadata from shard: {shard_path}")

    dataset = wds.WebDataset(shard_path, shardshuffle=False).decode()
    for sample in dataset:
        sample_count += 1
        if "json" in sample:
            metadata = sample["json"]
            if isinstance(metadata, bytes):
                metadata = json.loads(metadata)
            if "language" in metadata:
                languages.add(metadata["language"].get("ISO639-3", "unknown"))
            if "project_name" in metadata:
                projects.add(metadata["project_name"])

    return sample_count, languages, projects


def find_manifest(webdataset_path: str | Path) -> Path | None:
    """Look for manifest.json in webdataset_path or its parent directory."""
    webdataset_path = Path(webdataset_path).resolve()
    locations = [webdataset_path / "manifest.json", webdataset_path.parent / "manifest.json"]
    for manifest in locations:
        if manifest.exists():
            return manifest
    return None


def upload_to_huggingface(webdataset_path: str | Path, dataset_name: str, token: str | None = None) -> None:
    """
    Upload WebDataset shards (directly, preserving folders) to HuggingFace.

    Args:
        webdataset_path: Directory with WebDataset shards
        dataset_name: HuggingFace dataset repo
        token: HuggingFace API token
    """
    webdataset_path = Path(webdataset_path).resolve()
    print(f"Uploading from {webdataset_path} to HuggingFace repo {dataset_name}")

    shards = list(webdataset_path.rglob("*.tar"))
    if not shards:
        raise ValueError(f"No shards found in {webdataset_path}")

    print(f"Found {len(shards)} shards. Uploading in-place, preserving folders...")

    # Create README.md in-place
    readme_path = webdataset_path / "README.md"
    create_dataset_card(webdataset_path, readme_path, dataset_name)

    subfolder_names = [f.name for f in webdataset_path.iterdir() if f.is_dir()]

    # Copy manifest.json in-place if available
    # manifest_path = find_manifest(webdataset_path)
    # if manifest_path:
    #     print(f"Copying manifest.json to {webdataset_path}")
    #     (webdataset_path / "manifest.json").write_bytes(manifest_path.read_bytes())
    # else:
    #     print("⚠️ No manifest.json found, skipping.")

    print(f"Uploading to HuggingFace Hub as {dataset_name}")
    api = HfApi(token=token)

    try:
        upload_folder(
            folder_path=str(webdataset_path),
            repo_id=dataset_name,
            repo_type="dataset",
            token=token,
            ignore_patterns=["*.pyc", "__pycache__", ".git*"],
            commit_message="Upload dataset files",
            # delete_patterns=[
            #     f"{subfolder_name}/*.tar" for subfolder_name in subfolder_names
            # ],  # Overwrite any tar files already there, but leave other languages alone
        )
        print(f"✅ Successfully uploaded to {dataset_name}")
    except (HTTPError, HfHubHTTPError) as e:
        print(f"❌ HTTP error during upload: {e}")
        print("✔️ Suggestions:")
        print("- Check your HuggingFace token permissions")
        print("- Verify the dataset/organization name")
        print("- Confirm your network connection")
        raise
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        print("✔️ Suggestion: double-check your input arguments.")
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
    if not token:
        print("Warning: No HuggingFace API token provided. You may need to login interactively.")

    # Upload to HuggingFace
    upload_to_huggingface(args.webdataset_path, args.dataset_name, token)


if __name__ == "__main__":
    main()

# cd /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/upload_to_huggingface.py webdataset bible-nlp/sign-bibles
