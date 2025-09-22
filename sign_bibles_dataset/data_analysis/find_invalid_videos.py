#!/usr/bin/env python3
import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


@contextlib.contextmanager
def capture_stderr():
    """Temporarily capture stderr output (e.g., from FFmpeg inside OpenCV)."""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield sys.stderr
    finally:
        sys.stderr = old_stderr


def is_fully_readable(path: Path) -> bool:
    """
    Try to read all frames in a video file.
    Returns False if FFmpeg/OpenCV emit errors to stderr or if reading fails early.
    """
    with capture_stderr() as err_buf:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False

        while True:
            ret, _ = cap.read()
            if not ret:
                break

        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        ffmpeg_errors = err_buf.getvalue()

    if ffmpeg_errors.strip():
        return False

    return frame_pos == frame_count and frame_count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check validity of MP4 files in a directory.")
    parser.add_argument("directory", type=Path, help="Directory containing MP4 files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    mp4_files = list(args.directory.rglob("*.mp4"))
    logging.info("Found %d mp4 files in %s", len(mp4_files), args.directory)

    unreadable = []
    for mp4 in tqdm(mp4_files, desc="Checking videos"):
        if not is_fully_readable(mp4):
            logging.warning("Corrupted or incomplete video: %s", mp4)
            unreadable.append(mp4)
    report = "\n".join(str(mp4.resolve()) for mp4 in unreadable)
    logging.info(f"{report}")
    logging.info(f"Found {len(unreadable)} unreadable")


if __name__ == "__main__":
    main()
