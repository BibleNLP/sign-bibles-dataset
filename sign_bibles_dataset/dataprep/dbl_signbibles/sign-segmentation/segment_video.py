import os
import subprocess
import cv2
import pympi
import argparse
import boto3
from pathlib import Path
import tempfile


MIN_SEGMENT_DURATION_MS = 5000
GAP_SEGMENT_DURATION_MS = 15000


def download_from_s3(bucket_name, key, local_path):
    """Download a file from S3 to a local path."""
    s3_client = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3_client.download_file(bucket_name, key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False


def run_pose_extraction(input_video_path: Path, pose_file: Path):
    if pose_file.exists():
        print(f"{pose_file} already exists")
        return
    print(f"{pose_file} doesn't exist: estimating")
    subprocess.run(
        [
            "video_to_pose",
            "-i",
            str(input_video_path),
            "-o",
            str(pose_file),
            "--additional-config=model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true",
        ]
    )
    print("Pose extraction completed")


def get_annotations(eaf_file: Path):
    eaf = pympi.Elan.Eaf(str(eaf_file))
    print("Available tiers:", list(eaf.get_tier_names()))
    return sorted(
        eaf.get_annotation_data_for_tier("SENTENCE")
        if "SENTENCE" in eaf.get_tier_names()
        else [],
        key=lambda x: x[0],
    )


def get_video_duration_ms(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return int((total_frames / fps) * 1000)


def fill_gaps_and_merge(annotations, video_duration):
    segments = []
    current_time = 0

    for start, end, text in annotations:
        print(start, end, text)
        while current_time < start:
            gap_end = min(current_time + GAP_SEGMENT_DURATION_MS, start)
            segments.append((current_time, gap_end, ""))
            current_time = gap_end
        segments.append((start, end, text))
        current_time = end

    while current_time < video_duration:
        segment_end = min(current_time + GAP_SEGMENT_DURATION_MS, video_duration)
        segments.append((current_time, segment_end, ""))
        current_time = segment_end

    print(f"Created {len(segments)} segments")
    return merge_short_segments(segments)


def merge_short_segments(segments):
    merged = []
    i = 0

    while i < len(segments):
        start, end, text = segments[i]
        duration = end - start

        if duration >= MIN_SEGMENT_DURATION_MS:
            merged.append((start, end, text))
            i += 1
            continue

        # Try to merge with the next segment
        if i + 1 < len(segments):
            next_start, next_end, next_text = segments[i + 1]
            new_start = start
            new_end = next_end
            new_text = f"{text} {next_text}".strip()
            segments[i + 1] = (new_start, new_end, new_text)
        else:
            # Merge with previous if no next exists
            if merged:
                prev_start, prev_end, prev_text = merged.pop()
                merged_text = f"{prev_text} {text}".strip()
                merged.append((prev_start, end, merged_text))
            else:
                # Nothing to merge with, just keep it
                merged.append((start, end, text))
        i += 1

    print(f"Merged into {len(merged)} segments")
    return merged


def extract_segment(input_video_path, start_ms, end_ms, output_file):
    start_s = start_ms / 1000
    duration_s = (end_ms - start_ms) / 1000
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video_path),
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-force_key_frames",
        f"expr:gte(t,{start_s})",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(output_file),
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_file.exists() and output_file.stat().st_size > 10_000


def process_video(input_video_path):
    input_video_path = Path(input_video_path)
    output_dir = (input_video_path.parent / "segments").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_video_path.stem
    pose_file = input_video_path.with_suffix(".pose")
    eaf_file = input_video_path.with_suffix(".eaf")

    run_pose_extraction(input_video_path, pose_file)
    print("Segmentation completed")

    annotations = get_annotations(eaf_file)
    print(f"Found {len(annotations)} annotations")

    video_duration = get_video_duration_ms(input_video_path)
    segments = fill_gaps_and_merge(annotations, video_duration)

    output_pattern = output_dir / f"{base_name}_segment_%04d.mp4"
    valid_segments = []

    for i, (start_ms, end_ms, text) in enumerate(segments):
        duration = end_ms - start_ms
        if duration < MIN_SEGMENT_DURATION_MS:
            print(f"Segment {i} too short ({duration} ms), skipping")
            continue

        output_file = Path(str(output_pattern) % (len(valid_segments) + 1))
        if extract_segment(input_video_path, start_ms, end_ms, output_file):
            valid_segments.append((start_ms, end_ms, text))
            print(f"Created segment {len(valid_segments)}: {start_ms}–{end_ms}ms")
        else:
            print(f"Failed to create segment: {start_ms}–{end_ms}ms")

    print(f"Successfully created {len(valid_segments)} valid segments")


def process_input(input_source):
    """Process input source which can be a file, directory, or S3 bucket."""
    # Check if input is an S3 path (s3://bucket-name/key)
    if input_source.startswith("s3://"):
        # Parse S3 URL
        parts = input_source[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid S3 URL format. Use s3://bucket-name/key")

        bucket_name, key = parts
        # Create temporary directory for downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            if key.endswith(("/", "")):  # If it's a prefix (directory)
                s3_client = boto3.client("s3")
                paginator = s3_client.get_paginator("list_objects_v2")

                # List all objects with the given prefix
                for page in paginator.paginate(Bucket=bucket_name, Prefix=key):
                    for obj in page.get("Contents", []):
                        if obj["Key"].endswith((".mp4", ".avi", ".mov")):
                            local_path = os.path.join(
                                temp_dir, os.path.basename(obj["Key"])
                            )
                            if download_from_s3(bucket_name, obj["Key"], local_path):
                                print(f"\nProcessing {obj['Key']}...")
                                process_video(local_path)
            else:  # Single file
                local_path = os.path.join(temp_dir, os.path.basename(key))
                if download_from_s3(bucket_name, key, local_path):
                    print(f"\nProcessing {key}...")
                    process_video(local_path)

    # Local file or directory
    else:
        input_path = Path(input_source)
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_source}")

        if input_path.is_file():
            if input_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                print(f"\nProcessing {input_path.name}...")
                process_video(str(input_path))
            else:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")

        elif input_path.is_dir():
            for file_path in input_path.glob("*"):
                if file_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                    print(f"\nProcessing {file_path.name}...")
                    process_video(str(file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video files for sign language segmentation"
    )
    parser.add_argument(
        "input",
        help="Input source: can be a file path, directory path, or S3 URL (s3://bucket-name/key)",
    )
    args = parser.parse_args()

    process_input(args.input)
