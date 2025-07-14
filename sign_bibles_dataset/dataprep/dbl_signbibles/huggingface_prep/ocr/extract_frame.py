#!/usr/bin/env python3

import argparse
import sys
import tempfile
from pathlib import Path

import cv2


def extract_frame_to_temp(video_path: Path, frame_index: int) -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / f"{video_path.stem}.frame{frame_index}.png"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            sys.exit(f"Error: Cannot open video file {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= total_frames or frame_index < 0:
            cap.release()
            sys.exit(f"Error: Frame index {frame_index} out of range (total frames: {total_frames})")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            sys.exit(f"Error: Failed to read frame {frame_index} from {video_path}")

        cv2.imwrite(str(output_path), frame)
        print(output_path.resolve())  # Output the path to stdout

        input("Press Enter to close/delete file")  # Pause so file isn't deleted
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a specific frame from a video and save as PNG in a temporary directory."
    )
    parser.add_argument("video_path", type=Path, help="Path to the .mp4 video file.")
    parser.add_argument("frame_index", type=int, help="Frame index to extract.")

    args = parser.parse_args()
    extract_frame_to_temp(args.video_path, args.frame_index)
