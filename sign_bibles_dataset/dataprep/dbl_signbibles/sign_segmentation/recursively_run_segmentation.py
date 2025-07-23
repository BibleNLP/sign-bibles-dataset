#!/usr/bin/env python

import argparse
import logging
import time
from pathlib import Path

from pose_format import Pose
from sign_language_segmentation.bin import segment_pose
from tqdm import tqdm

MODEL_CHOICES = ["model_E4s-1.pth", "model_E1s-1.pth"]


def find_pose_files(search_path: Path):
    """Recursively find all .pose files under root_dir."""
    if search_path.is_file() and search_path.name.endswith(".pose"):
        return [search_path]
    return list(search_path.rglob("*.pose"))


def get_eaf_file(pose_file: Path, model: str) -> Path:
    """Determine the EAF file name corresponding to pose file and model."""
    model_name = Path(model).stem
    return pose_file.with_name(f"{pose_file.stem}.{model_name}.eaf")


def has_eaf(pose_file: Path, model: str) -> bool:
    return get_eaf_file(pose_file, model).exists()


def write_eaf_file(pose_file: Path, model: str, eaf) -> None:
    eaf_file = get_eaf_file(pose_file, model)
    eaf.to_file(eaf_file)
    logging.debug(f"Saved {eaf_file}")


def segment_pose_file(pose_file: Path, model: str, verbose=False):
    logging.debug(f"Processing {pose_file.name} with {model}...")
    with pose_file.open("rb") as f:
        pose = Pose.read(f)

    eaf, _ = segment_pose(pose, model=model, verbose=verbose)
    write_eaf_file(pose_file, model, eaf)


def recursively_run_segmentation(
    search_path: Path,
    model: str | None = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    models = [model] if model else MODEL_CHOICES
    pose_files = find_pose_files(search_path)
    logging.info(f"Found {len(pose_files)} pose files under {search_path}")

    total_start = time.perf_counter()
    skipped, processed = 0, 0
    model_timings = {}

    for model in models:
        model_start = time.perf_counter()
        for pose_file in tqdm(pose_files, desc=f"Segmenting with {model}"):
            eaf_file = get_eaf_file(pose_file, model)
            if eaf_file.exists() and not overwrite:
                skipped += 1
                logging.debug(f"Skipping {pose_file.name} with {model} (EAF exists).")
                continue
            segment_pose_file(pose_file, model, verbose=verbose)
            processed += 1
        model_elapsed = time.perf_counter() - model_start
        model_timings[model] = model_elapsed
        logging.info(f"Model {model} completed in {model_elapsed:.2f} seconds")

    total_elapsed = time.perf_counter() - total_start
    logging.info(f"Total time: {total_elapsed:.2f} seconds")
    logging.info(f"Processed {processed} files, skipped {skipped}")

    logging.info("Per-model timings:")
    for model, duration in model_timings.items():
        logging.info(f"  {model}: {duration:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Batch segmentation of pose files to EAF files.")
    parser.add_argument("search_path", type=Path, help="Root directory or pose file to process")
    parser.add_argument("--model", choices=MODEL_CHOICES, help=f"Model to use (default: all: {MODEL_CHOICES})")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    recursively_run_segmentation(
        search_path=args.search_path,
        model=args.model,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
