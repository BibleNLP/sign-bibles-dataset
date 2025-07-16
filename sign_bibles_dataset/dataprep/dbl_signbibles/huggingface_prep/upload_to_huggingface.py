#!/usr/bin/env python3
"""Upload WebDataset to HuggingFace datasets."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_large_folder
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import HTTPError
from tqdm import tqdm

import webdataset as wds
from sign_bibles_dataset.dataprep.dbl_signbibles.huggingface_prep.build_dataset_card import (
    build_dataset_card_markdown_and_yaml,
)

# TODO: add tasks like in https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/main/README.md or https://huggingface.co/datasets/racineai/OGC_MEGA_MultiDomain_DocRetrieval/blob/main/README.md or https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/README.md, maybe also text-to-video like https://huggingface.co/datasets/nkp37/OpenVid-1M


def find_shards_recursive(root_path: str | Path) -> list[str]:
    """Recursively find all .tar shards in the directory."""
    root = Path(root_path)
    shards = sorted(str(p) for p in root.rglob("*.tar"))
    if not shards:
        raise ValueError(f"No WebDataset shards found at {root}")
    return shards


def find_shards_by_subfolder(root_path: str | Path) -> dict[str, list[Path]]:
    """Find all .tar shards grouped by their immediate subfolder."""
    root = Path(root_path)
    shards_by_folder = {}
    for shard in root.rglob("*.tar"):
        subfolder = shard.parent.name
        shards_by_folder.setdefault(subfolder, []).append(shard)
    if not shards_by_folder:
        raise ValueError(f"No shards found in {root}")
    return shards_by_folder


def load_cached_stats(stats_path: Path, shard_count: int, subfolder_count: int) -> dict | None:
    if not stats_path.is_file():
        return None
    with open(stats_path) as f:
        cached = json.load(f)
    if cached.get("shard_count") == shard_count and cached.get("subfolder_count") == subfolder_count:
        print(f"✅ Using cached stats from {stats_path}")
        return cached
    print("⚠️ Cache is outdated, will recompute stats.")
    return None


def compute_stats_by_sampling(shards_by_folder: dict[str, list[Path]], sample_count: int | None = 1) -> dict:
    """
    Estimate dataset stats by sampling shards per subfolder.

    Args:
        shards_by_folder: Mapping of subfolder name to list of shard Paths.
        sample_count: Number of shards to sample per subfolder.
                      If None, use all shards.

    """
    total_samples = 0
    languages, projects = set(), set()
    language_stats = {}

    for subfolder, shards in tqdm(shards_by_folder.items(), desc="Sampling shards by subfolder"):
        language_stats[subfolder] = {"shard_count": len(shards)}
        sample_shards = shards if sample_count is None else shards[:sample_count]
        subfolder_sample_total, sampled_shards = 0, 0

        for shard in tqdm(sample_shards, desc=f"Subfolder: {subfolder}", disable=len(sample_shards) <= 1):
            count, langs, projs = parse_shard_metadata(str(shard))
            subfolder_sample_total += count
            languages.update(langs)
            projects.update(projs)
            sampled_shards += 1

        if sampled_shards > 0:
            estimated_total = (subfolder_sample_total / sampled_shards) * len(shards)
            total_samples += estimated_total

        language_stats[subfolder]["estimated_samples"] = estimated_total

    return {
        "total_samples_estimated": int(total_samples),
        "languages": sorted(languages),
        "projects": sorted(projects),
        "stats_per_language": language_stats,
    }


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
    locations = [
        webdataset_path / "manifest.json",
        webdataset_path.parent / "manifest.json",
        # Path.cwd() / "manifest.json",
    ]
    for manifest in locations:
        if manifest.exists():
            print(f"Found manifest at {manifest}")
            return manifest
    return None


def create_dataset_card(
    webdataset_path: str,
    output_path: str,
    dataset_name: str,
    language_codes: list[str] | None = None,
) -> str:
    webdataset_path = Path(webdataset_path)
    shards_by_folder = find_shards_by_subfolder(webdataset_path)

    total_shards = sum(len(s) for s in shards_by_folder.values())
    total_subfolders = len(shards_by_folder)
    stats_cache_path = webdataset_path / "dataset_stats.json"

    cached = load_cached_stats(stats_cache_path, total_shards, total_subfolders)

    if cached:
        total_samples = cached["total_samples_estimated"]
        languages = set(cached["languages"])
        projects = set(cached["projects"])
    else:
        stats = compute_stats_by_sampling(shards_by_folder)
        total_samples = stats["total_samples_estimated"]
        languages = set(stats["languages"])
        projects = set(stats["projects"])

        # cache new stats
        cache_data = {
            "subfolder_count": total_subfolders,
            "shard_count": total_shards,
            "total_samples_estimated": total_samples,
            "languages": sorted(languages),
            "projects": sorted(projects),
            "stats_by_language": stats["stats_per_language"],
        }
        with open(stats_cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"✅ Cached stats saved to {stats_cache_path}")

    if language_codes:
        languages.update(language_codes)

    card_text = build_dataset_card_markdown_and_yaml(
        dataset_name=dataset_name,
        shard_count=total_shards,
        total_sample_count=total_samples,
        languages=languages,
        projects=projects,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_text)

    print(f"✅ Dataset card written to {output_path}")
    return output_path


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

    # manifest_path = find_manifest(webdataset_path)
    # if manifest_path is not None:
    #     with open(manifest_path) as mf:
    #         manifest = json.load(mf)
    # else:
    #     manifest = None

    print(f"Found {len(shards)} shards. Uploading in-place, preserving folders...")

    # Create README.md in-place
    readme_path = webdataset_path / "README.md"
    create_dataset_card(webdataset_path, readme_path, dataset_name)

    # # Copy manifest.json in-place if available
    # if manifest_path:
    #     print(f"Copying manifest.json to {webdataset_path}")
    #     (webdataset_path / "manifest.json").write_bytes(manifest_path.read_bytes())
    # else:
    #     print("⚠️ No manifest.json found, skipping.")

    print(f"Uploading to HuggingFace Hub as {dataset_name}")
    api = HfApi(token=token)

    try:
        upload_large_folder(
            folder_path=str(webdataset_path),
            repo_id=dataset_name,
            repo_type="dataset",
            # token=token,
            ignore_patterns=["*.pyc", "__pycache__", ".git*", "manifest.json"],
            # commit_message="Upload dataset files",
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
