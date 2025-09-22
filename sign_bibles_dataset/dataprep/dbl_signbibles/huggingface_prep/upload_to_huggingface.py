#!/usr/bin/env python3
"""Upload WebDataset to HuggingFace datasets."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, upload_large_folder
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import HTTPError
from tqdm import tqdm

import webdataset as wds
from sign_bibles_dataset.dataprep.dbl_signbibles.huggingface_prep.build_dataset_card import (
    build_dataset_card_markdown_and_yaml,
)


def find_shards_by_subfolder(root_path: str | Path) -> dict[str, list[Path]]:
    """Group shards by their full relative subfolder (language/project)."""
    root = Path(root_path)
    shards_by_folder = defaultdict(list)
    for shard in root.rglob("*.tar"):
        relative_folder = shard.parent.relative_to(root).as_posix()  # language/project
        shards_by_folder[relative_folder].append(shard)

    if not shards_by_folder:
        raise ValueError(f"No shards found in {root.resolve()}")
    return shards_by_folder


def load_cached_stats(stats_path: Path, shard_count: int, subfolder_count: int) -> dict | None:
    if not stats_path.is_file():
        return None
    with open(stats_path) as f:
        cached = json.load(f)

    if cached.get("shard_count") == shard_count and cached.get("subfolder_count") == subfolder_count:
        print(f"✅ Using cached stats from {stats_path}")
        return cached

    print(f"⚠️ Cached stats outdated (found {stats_path}), recomputing.")
    return None


def compute_stats_by_sampling(shards_by_folder: dict[str, list[Path]], sample_count: int | None = 3) -> dict:
    total_samples = 0
    languages, projects = set(), set()
    stats_per_folder = {}
    configs = {}

    for folder, shards in tqdm(shards_by_folder.items(), desc="Sampling shards"):
        lang = Path(folder).parts[0]
        project = Path(folder).parts[1] if len(Path(folder).parts) > 1 else "unknown"

        lang_config_lines = []
        lang_config_lines.append(f"  - config_name: {lang.replace('/', '_')}\n    data_files:")
        project_config_lines = []
        project_config_lines.append(f"  - config_name: {project.replace('/', '_')}\n    data_files:")

        for split in ["train", "val", "test"]:
            lang_config_lines.append(f"      - split: {split}\n        path: {lang}/*/*{split}.tar")
            project_config_lines.append(f"      - split: {split}\n        path: {folder}/*{split}.tar")
        configs[lang] = lang_config_lines

        configs[project] = project_config_lines
        languages.add(lang)
        projects.add(project)

        sample_shards = shards if sample_count is None else shards[:sample_count]
        folder_samples = 0

        for shard in tqdm(sample_shards, desc=f"Sampling {folder}", leave=False):
            count, langs, projs = parse_shard_metadata(str(shard))
            folder_samples += count
            languages.update(langs)
            projects.update(projs)

        estimated = (folder_samples / len(sample_shards)) * len(shards) if sample_shards else 0
        total_samples += estimated

        stats_per_folder[folder] = {
            "shard_count": len(shards),
            "estimated_samples": int(estimated),
        }
        # print(json.dumps(configs, indent=2))

    return {
        "total_samples_estimated": int(total_samples),
        "languages": sorted(languages),
        "projects": sorted(projects),
        "stats_per_folder": stats_per_folder,
        "configs": configs,
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


def create_dataset_card(
    webdataset_path: str, output_path: str, dataset_name: str, language_codes: list[str] | None = None
) -> str:
    webdataset_path = Path(webdataset_path)
    shards_by_folder = find_shards_by_subfolder(webdataset_path)

    total_shards = sum(len(shards) for shards in shards_by_folder.values())
    total_subfolders = len(shards_by_folder)
    stats_cache_path = webdataset_path / "dataset_stats.json"

    cached = load_cached_stats(stats_cache_path, total_shards, total_subfolders)

    if cached:
        stats = cached
    else:
        stats = compute_stats_by_sampling(shards_by_folder)
        stats.update(
            {
                "shard_count": total_shards,
                "subfolder_count": total_subfolders,
            }
        )
        with stats_cache_path.open("w") as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Cached stats saved to {stats_cache_path}")

    languages = set(stats["languages"])
    if language_codes:
        languages.update(language_codes)

    card_text = build_dataset_card_markdown_and_yaml(
        dataset_name=dataset_name,
        shard_count=total_shards,
        total_sample_count=stats["total_samples_estimated"],
        languages=languages,
        projects=stats["projects"],
        configs=stats["configs"],
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_text)

    print(f"✅ Dataset card written to {output_path}")
    return output_path


def upload_to_huggingface(
    webdataset_path: str | Path, dataset_name: str, token: str | None = None, skip_readme=True
) -> None:
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

    if not skip_readme:
        # Create README.md in-place
        readme_path = webdataset_path / "README.md"
        create_dataset_card(webdataset_path, readme_path, dataset_name)
        print(f"README CREATED at {readme_path.resolve()}")

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

# cd /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/upload_to_huggingface.py webdataset_large_files_removed bible-nlp/sign-bibles

# cd /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/upload_to_huggingface.py /data/petabyte/cleong/data/DBL_Deaf_Bibles/temp/ bible-nlp/sign-bibles

# cd /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/upload_to_huggingface.py webdataset_large_files_removed bible-nlp/sign-bibles
