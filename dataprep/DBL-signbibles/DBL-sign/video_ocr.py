#!/usr/bin/env python3

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import pytesseract
from tqdm import tqdm


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int,
    min_frame: int,
    max_frame: int | None,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = max(min_frame, 0)
    end = min(max_frame if max_frame is not None else total_frames, total_frames)

    if start >= end:
        raise ValueError(
            f"Invalid frame range: min_frame={min_frame}, max_frame={max_frame}, total={total_frames}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_paths = []
    current_frame = start
    frames_to_process = end - start
    pbar = tqdm(total=frames_to_process, desc="Extracting frames")

    while current_frame < end:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_skip == 0 or (current_frame - start) % (frame_skip + 1) == 0:
            frame_path = output_dir / f"frame_{current_frame:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)

        current_frame += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    return frame_paths


def run_ocr_on_frames(frame_paths: list[Path]):
    for frame in tqdm(sorted(frame_paths), desc="Running OCR"):
        ocr_text = pytesseract.image_to_string(str(frame)).strip()
        # Get numeric frame index from filename like 'frame_03000.png'
        frame_index = int(frame.stem.split("_")[1])
        yield {"frame_index": frame_index, "frame_name": frame.name, "text": ocr_text}


def process_video(
    video_path: Path, frame_skip: int, min_frame: int, max_frame: int | None
) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"Extracting frames to temporary directory: {tmp_path}")
        frame_paths = extract_frames(
            video_path, tmp_path, frame_skip, min_frame, max_frame
        )
        print(f"Extracted {len(frame_paths)} frames.")
        return list(run_ocr_on_frames(frame_paths))


def main():
    parser = argparse.ArgumentParser(description="Run OCR on video frames.")
    parser.add_argument("video", type=Path, help="Path to the .mp4 video file")
    parser.add_argument("--out", type=Path, help="Path to save OCR results as JSON")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Skip every N frames (default: 1 = every other frame)",
    )
    parser.add_argument(
        "--min-frame",
        type=int,
        default=0,
        help="Start processing at this frame index (default: 0)",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        help="Stop processing before this frame index (default: end of video)",
    )

    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"Video file not found: {args.video}")

    if Path(args.video).is_dir():
        videos = list(args.video.glob("*.mp4"))
    else:
        videos = [args.video]

    if args.frame_skip < 0:
        parser.error("Frame skip must be 0 or greater.")
    if args.min_frame < 0:
        parser.error("min-frame must be >= 0.")
    if args.max_frame is not None and args.max_frame <= args.min_frame:
        parser.error("max-frame must be greater than min-frame.")

    for video in tqdm(videos, desc="Processing Videos"):
        ocr_data = process_video(video, args.frame_skip, args.min_frame, args.max_frame)

        out_path = args.out
        if out_path is None:
            out_path = video.with_name(video.stem + "_ocrtext.json")

        print(f"Writing OCR results to {out_path}")
        out_path.write_text(
            json.dumps(ocr_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
