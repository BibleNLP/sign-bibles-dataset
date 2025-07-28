import argparse
import logging
import re
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from brisque import BRISQUE
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

warnings.filterwarnings("ignore")  # Suppress Python warnings about invalid values


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def timed_section(name: str):
    start = time.perf_counter()
    result = {}
    yield result
    end = time.perf_counter()
    duration = end - start
    result["duration"] = duration
    logger.info(f"[TIMER] {name} took {duration:.2f} seconds.")


def brisque_scores_from_video_frames(video_path: Path, overwrite: bool = False) -> pd.DataFrame:
    """
    Given a path to an MP4 file, find the corresponding frame directory named `<video.stem>_frames`,
    apply BRISQUE to each PNG frame (non-recursively), and return a DataFrame with frame numbers and scores.
    If a cached parquet file exists and `overwrite` is False, it is loaded and returned instead.

    Parameters
    ----------
    video_path : Path
        Path to the input video file.
    overwrite : bool
        Whether to recompute BRISQUE scores if the cached parquet exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['frame', 'score'], sorted by frame number.

    """
    frame_dir = video_path.parent / f"{video_path.stem}_frames"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Expected frame directory: {frame_dir}")

    parquet_path = frame_dir / f"{video_path.stem}.brisque_scores.parquet"
    if parquet_path.exists() and not overwrite:
        logger.debug(f"Using cached BRISQUE scores at {parquet_path}")
        return pd.read_parquet(parquet_path)

    brisque = BRISQUE(url=False)
    frame_pattern = re.compile(r"frame_(\d+)\.png")

    results = []

    for img_path in sorted(frame_dir.glob("*.png")):
        match = frame_pattern.fullmatch(img_path.name)
        if not match:
            continue  # Skip files that don't match expected pattern

        frame_num = int(match.group(1))

        try:
            img = Image.open(img_path)
            img_array = np.asarray(img)
        except (OSError, UnidentifiedImageError) as e:
            logger.debug(f"Failed to load image {img_path}: {e}")
            continue

        try:
            with timed_section("Run BRISQUE score") as duration:
                score = brisque.score(img=img_array)

        except ValueError as e:
            logger.debug(f"Failed to compute BRISQUE score for {img_path}: {e}")
            continue

        results.append((frame_num, score, duration["duration"]))

    df = (
        pd.DataFrame(results, columns=["frame", "score", "score_duration_sec"])
        .sort_values("frame")
        .reset_index(drop=True)
    )

    df.to_parquet(parquet_path, index=False)
    logger.info(f"Mean BRISQUE for video ({len(df)} frames): {df['score'].mean()}")
    logger.info(f"Saved BRISQUE scores to {parquet_path}")

    return df


def brisque_scores_from_folder(folder: Path, recursive: bool = False, overwrite: bool = False) -> pd.DataFrame:
    """
    Compute BRISQUE scores for all `*.mp4` files in a folder.

    For each video, looks for a corresponding frame folder named `<stem>_frames`,
    runs BRISQUE on all matching PNGs inside it, and collects results into a single DataFrame.

    Videos without frame folders are silently skipped but counted.

    Parameters
    ----------
    folder : Path
        Directory containing .mp4 files.
    recursive : bool
        Whether to search subdirectories recursively.
    overwrite : bool
        Whether to overwrite cached parquet files in frame folders.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns ['video', 'frame', 'score'].

    """
    video_paths = sorted(folder.rglob("*.mp4") if recursive else folder.glob("*.mp4"))

    all_results = []
    skipped = 0
    processed = 0

    for video_path in tqdm(video_paths, desc="Processing videos"):
        try:
            df = brisque_scores_from_video_frames(video_path, overwrite=overwrite)
        except FileNotFoundError:
            skipped += 1
            continue

        if not df.empty:
            df.insert(0, "video", video_path.name)
            all_results.append(df)
            processed += 1

    logger.info(f"BRISQUE summary: processed {processed}, skipped {skipped} due to missing frame folders.")

    return (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame(columns=["video", "frame", "score", "score_duration_sec"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute BRISQUE scores from video frame folders or individual .mp4 files."
    )
    parser.add_argument("path", type=Path, help="Either a directory containing .mp4 files or a single .mp4 file")
    parser.add_argument(
        "--recursive", action="store_true", help="Search for .mp4 files recursively (only applies to directories)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet files in _frames folders")
    args = parser.parse_args()
    print(args)

    path = args.path

    if path.is_file() and path.suffix.lower() == ".mp4":
        try:
            df = brisque_scores_from_video_frames(path, overwrite=args.overwrite)
        except FileNotFoundError:
            logger.warning(f"No frame folder found for {path.name}")
            return

        if df.empty:
            logger.warning(f"No BRISQUE scores computed for {path.name}")
            return

        avg = df["score"].mean()
        logger.info(f"Average BRISQUE score for {path.name}: {avg:.2f}")

    elif path.is_dir():
        df = brisque_scores_from_folder(path, recursive=args.recursive, overwrite=args.overwrite)

        if df.empty:
            logger.warning("No BRISQUE scores were computed.")
            return

        avg_scores = df.groupby("video")["score"].mean().reset_index()
        logger.info("Average BRISQUE score per video:")
        for _, row in avg_scores.iterrows():
            logger.info(f"{row['video']}: {row['score']:.2f}")

        overall_avg = avg_scores["score"].mean()
        logger.info(f"Overall average BRISQUE score across all videos: {overall_avg:.2f}")
    else:
        logger.error(f"Invalid path: {path}. Must be a directory or a .mp4 file.")


if __name__ == "__main__":
    main()
