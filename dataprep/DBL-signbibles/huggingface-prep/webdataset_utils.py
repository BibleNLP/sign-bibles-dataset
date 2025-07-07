#!/usr/bin/env python3
"""
Utility functions for working with WebDataset format for sign language videos.
"""

import os
import json
import tempfile
import webdataset as wds
from torch.utils.data import DataLoader
import cv2
import numpy as np


def load_webdataset(path, batch_size=1, shuffle=False, num_workers=4):
    """
    Load a WebDataset from a path.

    Args:
        path: Path to WebDataset shard or directory containing shards
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for DataLoader

    Returns:
        DataLoader for the WebDataset
    """
    # If path is a directory, find all .tar files
    if os.path.isdir(path):
        shards = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tar")]
        shards.sort()
    else:
        shards = [path]

    if not shards:
        raise ValueError(f"No WebDataset shards found at {path}")

    # Create dataset
    dataset = (
        wds.WebDataset(shards)
        .decode()
        .to_tuple("original.mp4", "pose.mp4", "mask.mp4", "segmentation.mp4", "json")
    )

    # Create DataLoader
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return loader


def extract_sample(sample, output_dir=None):
    """
    Extract a sample from a WebDataset to files.

    Args:
        sample: Sample from WebDataset
        output_dir: Directory to extract files to (if None, uses a temporary directory)

    Returns:
        Dictionary with paths to extracted files
    """
    original, pose, mask, segmentation, metadata_json = sample

    # Parse metadata
    metadata = json.loads(metadata_json)
    segment_name = metadata.get("segment_name", f"segment_{os.urandom(4).hex()}")

    # Create output directory
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, segment_name)

    os.makedirs(output_dir, exist_ok=True)

    # Save files
    extracted_files = {}

    if original is not None:
        original_path = os.path.join(output_dir, f"{segment_name}.original.mp4")
        with open(original_path, "wb") as f:
            f.write(original)
        extracted_files["original"] = original_path

    if pose is not None:
        pose_path = os.path.join(output_dir, f"{segment_name}.pose.mp4")
        with open(pose_path, "wb") as f:
            f.write(pose)
        extracted_files["pose"] = pose_path

    if mask is not None:
        mask_path = os.path.join(output_dir, f"{segment_name}.mask.mp4")
        with open(mask_path, "wb") as f:
            f.write(mask)
        extracted_files["mask"] = mask_path

    if segmentation is not None:
        segmentation_path = os.path.join(output_dir, f"{segment_name}.segmentation.mp4")
        with open(segmentation_path, "wb") as f:
            f.write(segmentation)
        extracted_files["segmentation"] = segmentation_path

    # Save metadata
    metadata_path = os.path.join(output_dir, f"{segment_name}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    extracted_files["metadata"] = metadata_path

    return extracted_files


def create_preview_video(sample, output_path, max_frames=300):
    """
    Create a preview video showing original, pose, mask, and segmentation side by side.

    Args:
        sample: Sample from WebDataset
        output_path: Path to save preview video
        max_frames: Maximum number of frames to include

    Returns:
        Path to preview video
    """
    # Extract sample to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        files = extract_sample(sample, temp_dir)

        # Check if we have all the necessary files
        if "original" not in files:
            raise ValueError("Original video not found in sample")

        # Open video files
        caps = {}
        for key, path in files.items():
            if key == "metadata":
                continue
            caps[key] = cv2.VideoCapture(path)
            if not caps[key].isOpened():
                print(f"Warning: Could not open {key} video")
                caps[key].release()
                del caps[key]

        if not caps:
            raise ValueError("No valid video files found in sample")

        # Get video properties from original
        width = int(caps["original"].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps["original"].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps["original"].get(cv2.CAP_PROP_FPS)

        # Calculate output dimensions
        num_videos = len(caps)
        output_width = width * num_videos
        output_height = height

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        # Read and combine frames
        frame_count = 0
        while frame_count < max_frames:
            # Read frames from all videos
            frames = {}
            for key, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    break
                frames[key] = frame

            # If any video has ended, break
            if len(frames) < len(caps):
                break

            # Combine frames
            combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

            # Add labels to frames
            for i, key in enumerate(caps.keys()):
                frame = frames[key].copy()
                cv2.putText(
                    frame, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                combined_frame[:, i * width : (i + 1) * width] = frame

            # Write frame
            out.write(combined_frame)
            frame_count += 1

        # Release resources
        for cap in caps.values():
            cap.release()
        out.release()

        return output_path


def list_webdataset_contents(path):
    """
    List the contents of a WebDataset.

    Args:
        path: Path to WebDataset shard or directory containing shards

    Returns:
        List of dictionaries with sample information
    """
    # If path is a directory, find all .tar files
    if os.path.isdir(path):
        shards = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tar")]
        shards.sort()
    else:
        shards = [path]

    if not shards:
        raise ValueError(f"No WebDataset shards found at {path}")

    # List contents
    contents = []

    for shard in shards:
        dataset = wds.WebDataset(shard).decode()

        for sample in dataset:
            key = sample.get("__key__", "unknown")

            # Parse metadata if available
            metadata = {}
            if "json" in sample:
                try:
                    metadata = json.loads(sample["json"])
                except:
                    pass

            # Get available components
            components = [k for k in sample.keys() if k not in ["__key__", "json"]]

            contents.append(
                {
                    "key": key,
                    "shard": os.path.basename(shard),
                    "components": components,
                    "metadata": metadata,
                }
            )

    return contents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WebDataset utilities for sign language videos"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List contents of a WebDataset")
    list_parser.add_argument(
        "path", help="Path to WebDataset shard or directory containing shards"
    )

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract samples from a WebDataset"
    )
    extract_parser.add_argument(
        "path", help="Path to WebDataset shard or directory containing shards"
    )
    extract_parser.add_argument("--output-dir", help="Directory to extract files to")
    extract_parser.add_argument(
        "--key",
        help="Key of sample to extract (if not specified, extracts all samples)",
    )

    # Preview command
    preview_parser = subparsers.add_parser(
        "preview", help="Create preview videos from a WebDataset"
    )
    preview_parser.add_argument(
        "path", help="Path to WebDataset shard or directory containing shards"
    )
    preview_parser.add_argument(
        "--output-dir", help="Directory to save preview videos to"
    )
    preview_parser.add_argument(
        "--key",
        help="Key of sample to preview (if not specified, previews first sample)",
    )
    preview_parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to include in preview",
    )

    args = parser.parse_args()

    if args.command == "list":
        contents = list_webdataset_contents(args.path)
        print(f"Found {len(contents)} samples in {args.path}")
        for i, sample in enumerate(contents):
            print(f"Sample {i + 1}: {sample['key']}")
            print(f"  Shard: {sample['shard']}")
            print(f"  Components: {', '.join(sample['components'])}")
            if "segment_name" in sample["metadata"]:
                print(f"  Segment: {sample['metadata']['segment_name']}")
            if (
                "language_code" in sample["metadata"]
                and "project_name" in sample["metadata"]
            ):
                print(
                    f"  Source: {sample['metadata']['language_code']}/{sample['metadata']['project_name']}"
                )
            print()

    elif args.command == "extract":
        # Load dataset
        loader = load_webdataset(args.path)

        # Create output directory
        output_dir = args.output_dir or "extracted_samples"
        os.makedirs(output_dir, exist_ok=True)

        # Extract samples
        for i, sample in enumerate(loader):
            # Unpack batch
            original, pose, mask, segmentation, metadata_json = [
                item[0] for item in sample
            ]
            metadata = json.loads(metadata_json)
            key = metadata.get("segment_name", f"sample_{i}")

            # Skip if not the requested key
            if args.key and key != args.key:
                continue

            # Extract sample
            sample_output_dir = os.path.join(output_dir, key)
            extracted_files = extract_sample(
                (original, pose, mask, segmentation, metadata_json), sample_output_dir
            )

            print(f"Extracted sample {key} to {sample_output_dir}")
            for file_type, file_path in extracted_files.items():
                print(f"  {file_type}: {os.path.basename(file_path)}")

            # If extracting a specific key, stop after finding it
            if args.key:
                break

    elif args.command == "preview":
        # Load dataset
        loader = load_webdataset(args.path)

        # Create output directory
        output_dir = args.output_dir or "preview_videos"
        os.makedirs(output_dir, exist_ok=True)

        # Create preview videos
        for i, sample in enumerate(loader):
            # Unpack batch
            original, pose, mask, segmentation, metadata_json = [
                item[0] for item in sample
            ]
            metadata = json.loads(metadata_json)
            key = metadata.get("segment_name", f"sample_{i}")

            # Skip if not the requested key
            if args.key and key != args.key:
                continue

            # Create preview video
            preview_path = os.path.join(output_dir, f"{key}_preview.mp4")
            create_preview_video(
                (original, pose, mask, segmentation, metadata_json),
                preview_path,
                args.max_frames,
            )

            print(f"Created preview video for sample {key}: {preview_path}")

            # If previewing a specific key or no key specified (just preview first sample), stop
            if args.key or not args.key:
                break
