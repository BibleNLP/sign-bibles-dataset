#!/usr/bin/env nix-shell
#!nix-shell -i python -p "python3.withPackages (p: with p; [ (opencv4.override { enableGtk3 = true; }) pytesseract numpy ])"

import argparse
import subprocess
from pathlib import Path

import cv2
import pandas as pd
import pytesseract
from tqdm import tqdm


def get_roi_coordinates(frame_shape, position="top_right", roi_fraction=(0.2, 0.3)):
    height, width = frame_shape[:2]
    h_frac, w_frac = roi_fraction

    h_size = int(height * h_frac)
    w_size = int(width * w_frac)

    if position == "top_left":
        return (0, 0, w_size, h_size)
    elif position == "top_right":
        return (width - w_size, 0, width, h_size)
    elif position == "bottom_left":
        return (0, height - h_size, w_size, height)
    elif position == "bottom_right":
        return (width - w_size, height - h_size, width, height)
    elif position == "center":
        return (
            (width - w_size) // 2,
            (height - h_size) // 2,
            (width + w_size) // 2,
            (height + h_size) // 2,
        )
    else:
        raise ValueError(f"Unsupported ROI position: {position}")


def process_video(video_path, output_dir: Path, visualize: bool, roi_position="top_right"):
    output_dir.mkdir(exist_ok=True, parents=True)
    previous_text = None
    timestamps = []
    texts = []
    frame_indices = []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0

    iterator = range(total_frames)
    if total_frames > 1000:
        iterator = tqdm(iterator, desc="Processing Frames", unit="frame")

    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        # Get ROI coordinates
        x1, y1, x2, y2 = get_roi_coordinates(frame.shape, roi_position)
        roi = frame[y1:y2, x1:x2]

        # Draw rectangle if visualize is enabled
        if visualize:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()

        if text != previous_text:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamps.append(timestamp)
            texts.append(text)
            frame_indices.append(frame_index)
            previous_text = text

            if visualize:
                cv2.imwrite(
                    output_dir / f"TextChange_roi{roi_position}_frame{frame_index}_timestamp{timestamp:.2f}.png",
                    frame,
                )

        if frame_index % 100 == 0 and visualize:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            cv2.imwrite(
                output_dir / f"Video_frame_roi_{roi_position}_frame{frame_index}_timestamp{timestamp:.2f}.png",
                frame,
            )

        frame_index += 1

    cap.release()

    timestamps.append(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    texts.append(texts[-1])
    frame_indices.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Timestamps for cuts:", timestamps)

    timestamps_df = pd.DataFrame({"timestamp": timestamps, "text": texts, "frame_index": frame_indices})

    return timestamps_df


def cut_video(video_path, timestamps_df, output_dir):
    timestamps = timestamps_df["timestamp"].to_list()

    for i in range(len(timestamps) - 1):
        start_time = timestamps[i]
        end_time = timestamps[i + 1]
        output_file = output_dir / f"{video_path.stem}.segment_{i + 1}.mp4"

        # Construct FFmpeg command
        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c",
            "copy",
            str(output_file),
        ]

        # Execute the command
        subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a video based on text changes.")
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the video file to process, or dir containing .mp4 files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the video segments. Default: 'segments' folder in video path dir",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the region of interest in the video.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="top_right",
        choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"],
        help="Region of interest position in the frame.",
    )
    parser.add_argument(
        "--cut-video",
        action="store_true",
        help="Cut the video into segments. Will look for {video_path.stem}.text_change_timestamps.csv",
    )

    args = parser.parse_args()

    input_path = Path(args.video_path)
    video_paths = []
    if input_path.is_file():
        video_paths.append(input_path)
    if input_path.is_dir():
        video_paths = list(input_path.glob("*Passage*.mp4"))
        video_paths = [p for p in video_paths if ".pose-animation.mp4" not in p.name]

    for video_path in tqdm(video_paths, desc="Processing videos"):
        if args.output_dir is None:
            video_output_dir = video_path.parent / "segments" / video_path.stem
        else:
            video_output_dir = args.output_dir

        csv_path = video_output_dir / f"{video_path.stem}.text_change_timestamps.csv"
        if not csv_path.is_file():
            timestamps_df = process_video(video_path, video_output_dir, args.visualize, roi_position=args.roi)
            timestamps_df.to_csv(
                csv_path,
                index=False,
            )

        if args.cut_video:
            timestamps_df = pd.read_csv(csv_path)
            cut_video(video_path, timestamps_df, video_output_dir)
