import argparse
import logging
from pathlib import Path

import pandas as pd
import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> bool:
    """
    Extracts a frame from the video at a given index and saves it as an image.

    Args:
        video_path: Path to the video file.
        frame_index: Index of the frame to extract.
        output_path: Path to save the extracted frame.

    Returns:
        True if the frame was successfully extracted and saved, False otherwise.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= total_frames:
        logging.warning(
            f"Frame index {frame_index} out of range for {video_path.name} (total: {total_frames})"
        )
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logging.warning(f"Failed to read frame {frame_index} from {video_path.name}")
        return False

    success = cv2.imwrite(str(output_path), frame)
    if not success:
        logging.error(f"Failed to save frame {frame_index} to {output_path}")
    return success


def process_manualedit_csv(csv_path: Path) -> None:
    """
    Given a manualedit CSV file, extract all frames listed in 'frame_index'
    from the corresponding video.

    Args:
        csv_path: Path to the .ocr.manualedit.csv file.
    """
    base = csv_path.stem.replace(".ocr.manualedit", "")
    video_path = csv_path.with_name(f"{base}.mp4")

    if not video_path.exists():
        logging.warning(f"Skipping: Video not found for {csv_path.name}")
        return

    try:
        df = pd.read_csv(csv_path, usecols=["frame_index"])
    except Exception as e:
        logging.error(f"Could not read {csv_path}: {e}")
        return

    output_dir = csv_path.parent
    frame_indices = df["frame_index"].dropna().astype(int).unique()

    logging.info(
        f"Processing {csv_path.name}: extracting {len(frame_indices)} frames from {video_path.name}"
    )

    for idx in frame_indices:
        output_path = output_dir / f"{base}.frame{idx}.png"
        if output_path.exists():
            logging.debug(f"Thumbnail already exists: {output_path.name}")
            continue

        if extract_frame(video_path, idx, output_path):
            logging.info(f"Saved: {output_path.name}")


def main(directory: Path):
    if not directory.is_dir():
        logging.error(f"Not a directory: {directory}")
        return

    manualedit_files = sorted(directory.rglob("*.ocr.manualedit.csv"))
    logging.info(f"Found {len(manualedit_files)} '*.ocr.manualedit.csv' files.")

    for csv_path in manualedit_files:
        process_manualedit_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract thumbnails from .mp4 files using frame indices listed in .ocr.manualedit.csv files."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to search for *.ocr.manualedit.csv files recursively.",
    )
    args = parser.parse_args()
    main(args.directory)
