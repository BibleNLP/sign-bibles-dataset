#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from pose_format import Pose
from sign_language_segmentation.bin import segment_pose
from tqdm import tqdm


def find_pose_files(search_path: Path):
    """Recursively find all .pose files under root_dir."""
    if search_path.is_file() and search_path.name.endswith(".pose"):
        return [search_path]
    return list(search_path.rglob("*.pose"))


def has_eaf(pose_file: Path) -> bool:
    """Check if corresponding .eaf file exists."""
    eaf_file = pose_file.with_suffix(".eaf")
    return eaf_file.exists()


def write_eaf_file(pose_file: Path, eaf) -> None:
    eaf_file = pose_file.with_suffix(".eaf")
    eaf.to_file(eaf_file)
    logging.debug(f"Saved {eaf_file}")


def process_pose_file(pose_file: Path, model: str, verbose=False):
    logging.debug(f"Processing {pose_file}...")

    with pose_file.open("rb") as f:
        pose = Pose.read(f)

    eaf, _ = segment_pose(pose, model=model, verbose=verbose)

    # Always link pose
    eaf.add_linked_file(str(pose_file), mimetype="application/pose")

    write_eaf_file(pose_file, eaf)


def main():
    parser = argparse.ArgumentParser(description="Batch segmentation of pose files to EAF files.")
    parser.add_argument(
        "search_path", type=Path, help="Root directory to search for .pose files, or path to a .pose file"
    )
    parser.add_argument("--model", default="model_E4s-1.pth", help="Path to model (default model_E1s-1.pth)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    pose_files = find_pose_files(args.search_path)

    logging.info(f"Found {len(pose_files)} pose files under {args.search_path}")

    skipped, processed = 0, 0

    for pose_file in tqdm(pose_files, desc="Segmenting"):
        if has_eaf(pose_file) and not args.overwrite:
            skipped += 1
            continue
        try:
            process_pose_file(pose_file, model=args.model, verbose=args.verbose)
            processed += 1
        except Exception as e:
            logging.error(f"Failed processing {pose_file}: {e}")

    logging.info(f"Processing complete: {processed} files segmented, {skipped} skipped (already had EAF).")


if __name__ == "__main__":
    main()
