#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
1. Downloads videos from DBL-sign
2. Processes them with sign-segmentation
3. Packages everything into WebDataset format
"""

from pathlib import Path
import os
import sys
import json
import argparse
import shutil
import cv2
import requests
import time
import importlib.util
import tarfile
import random
import io
from processing_logger import ProcessingLogger

# Initialize the logger with the correct path
logger = ProcessingLogger(log_file_path="./output/run_log.txt")
# Clear the log at the start of the process
logger.clear_log()

# Add command line info to the log
logger.log_info(f"Command: {' '.join(sys.argv)}")


# Add parent directory to path for importing from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
dbl_sign_dir = os.path.join(parent_dir, "DBL-sign")
sign_seg_dir = os.path.join(parent_dir, "sign-segmentation")
sys.path.append(dbl_sign_dir)
sys.path.append(sign_seg_dir)

# Import DBL-sign utilities
# Import manifest generator functions - using a different import approach
# since the file has a dash in the name
manifest_generator_path = os.path.join(dbl_sign_dir, "DBL-manifest-generator.py")
manifest_generator_spec = importlib.util.spec_from_file_location(
    "DBL_manifest_generator", manifest_generator_path
)
manifest_generator = importlib.util.module_from_spec(manifest_generator_spec)
manifest_generator_spec.loader.exec_module(manifest_generator)


# Import sign-segmentation modules dynamically
def import_module_from_file(module_name, file_path):
    """
    Import a module from a file path.

    Args:
        module_name: Name to assign to the module
        file_path: Path to the Python file

    Returns:
        Imported module
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: Module file not found: {file_path}")
            return None

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Error: Failed to create module spec for {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        if module is None:
            print(f"Error: Failed to create module from spec for {file_path}")
            return None

        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error executing module {module_name} from {file_path}: {str(e)}")
            return None

        return module
    except Exception as e:
        print(f"Error importing module {module_name} from {file_path}: {str(e)}")
        return None


class DBLSignDownloader:
    """Class to handle downloading videos from DBL-sign."""

    def __init__(self, output_dir):
        """Initialize the downloader with output directory."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_path = os.path.join(dbl_sign_dir, "manifest.json")
        self.manifest = None

        # Load manifest
        self.load_manifest()

    def generate_fresh_manifest(self):
        """Generate a fresh manifest using the imported manifest generator functions."""
        print("Generating fresh manifest...")

        # Response data containing the list of sign language projects (hardcoded in the original script)
        response_data = {
            "aaData": [
                [
                    "245cca6ac5984130",
                    "Sri Lankan Sign Language",
                    "sqs",
                    "Sri Lanka",
                    "Chronological Bible Translation in Sri Lankan Sign Language",
                    "DOOR International",
                ],
                [
                    "055195093e2347d0",
                    "Burundian Sign Language",
                    "lsb",
                    "Burundi",
                    "Chronological Bible Translation in Burundian Sign Language",
                    "DOOR International",
                ],
                [
                    "0a3ea85ee1e34a2d",
                    "Nepali Sign Language",
                    "nsp",
                    "Nepal",
                    "Chronological Bible Translation in Nepali Sign Language",
                    "DOOR International",
                ],
                [
                    "1fcac35962494d40",
                    "Ugandan Sign Language",
                    "ugn",
                    "Uganda",
                    "Chronological Bible Translation in Ugandan Sign Language",
                    "DOOR International",
                ],
                [
                    "a63aef8004db401b",
                    "Ethiopian Sign Language",
                    "eth",
                    "Ethiopia",
                    "Chronological Bible Translation in Ethiopian Sign Language",
                    "DOOR International",
                ],
                [
                    "56c922b7b5a44a47",
                    "Kerala Sign Language",
                    "mis",
                    "India",
                    "Chronological Bible Translation in Kerala Sign Language",
                    "DOOR International",
                ],
                [
                    "65c350c1cf9c42e4",
                    "Nigerian Sign Language",
                    "nsi",
                    "Federal Republic Nigeria",
                    "Chronological Bible Translation in Nigerian Sign Language",
                    "DOOR International",
                ],
                [
                    "6ad9cf57ab084a3b",
                    "Estonian Sign Language",
                    "eso",
                    "Estonia",
                    "The Bible in Estonian Sign Language",
                    "Deaf Bible Society",
                ],
                [
                    "9e9e20d036fa4e91",
                    "Tanzanian Sign Language",
                    "tza",
                    "Tanzania",
                    "Chronological Bible Translation in Tanzanian Sign Language",
                    "DOOR International",
                ],
                [
                    "6c1ffbf874d14ee1",
                    "Andhra Pradesh Sign Language",
                    "mis",
                    "India",
                    "Chronological Bible Translation in Andhra Pradesh Sign Language",
                    "DOOR International",
                ],
                [
                    "2d6ac5c8b4614955",
                    "Bulgarian Sign Language",
                    "bqn",
                    "Bulgaria",
                    "Chronological Bible Translation in Bulgarian Sign Language",
                    "DOOR International",
                ],
                [
                    "1bacaede20da4494",
                    "American Sign Language",
                    "ase",
                    "United States of America",
                    "Chronological Bible Translation in American Sign Language (119 Introductions and Passages expanded with More Information)",
                    "Deaf Harbor ",
                ],
                [
                    "d2027facd4cc4c2a",
                    "American Sign Language",
                    "ase",
                    "United States of America",
                    "Chronological Bible Translation in American Sign Language (119 Introductions and Passages)",
                    "Deaf Harbor ",
                ],
                [
                    "6543fec2ced7421d",
                    "South Sudanese Sign Language",
                    "mis",
                    "Republic of South Sudan",
                    "Chronological Bible Translation in South Sudanese Sign Language",
                    "DOOR International",
                ],
                [
                    "a28def50f139432a",
                    "Kenyan Sign Language",
                    "xki",
                    "Kenya, Republic of",
                    "Chronological Bible Translation in Kenyan Sign Language",
                    "DOOR International",
                ],
                [
                    "c4b68657ce9b48ad",
                    "Ghanaian Sign Language",
                    "gse",
                    "Ghana",
                    "Chronological Bible Translation in Ghanaian Sign Language",
                    "DOOR International",
                ],
                [
                    "995240c9d7e8453e",
                    "Indian (Delhi) Sign Language",
                    "ins",
                    "India",
                    "Chronological Bible Translation in Indian (Delhi)  Sign Language",
                    "DOOR International",
                ],
                [
                    "6d5944a5ceb944c0",
                    "Russian Sign Language",
                    "rsl",
                    "Russian Federation",
                    "Chronological Bible Translation in Russian Sign Language",
                    "DOOR International",
                ],
                [
                    "ec8517dba29d4d93",
                    "Egyptian Sign Language",
                    "esl",
                    "Egypt",
                    "Chronological Bible Translation in Egyptian Sign Language",
                    "DOOR International",
                ],
                [
                    "c0b48facec324e4b",
                    "Mozambican Sign Language",
                    "mzy",
                    "Republic of Mozambique",
                    "Chronological Bible Translation in Mozambican Sign Language",
                    "DOOR International",
                ],
                [
                    "b963267b41cc443c",
                    "West Bengal Sign Language",
                    "mis",
                    "India",
                    "Chronological Bible Translation in West Bengal Sign Language",
                    "DOOR International",
                ],
            ]
        }

        # Create manifest using the imported function
        manifest = manifest_generator.create_manifest(response_data)

        # Save manifest using the imported function
        manifest_generator.save_manifest(manifest, self.manifest_path)

        print("Fresh manifest generated successfully.")

        # Load the newly generated manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        return self.manifest

    def load_manifest(self):
        """Load the manifest file."""
        if not os.path.exists(self.manifest_path):
            print("Manifest file not found. Generating a fresh one...")
            return self.generate_fresh_manifest()

        # Check if manifest is older than 24 hours
        manifest_age = time.time() - os.path.getmtime(self.manifest_path)
        if manifest_age > 86400:  # 24 hours in seconds
            print("Manifest is older than 24 hours. Generating a fresh one...")
            return self.generate_fresh_manifest()

        print("Loading existing manifest...")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        return self.manifest

    def refresh_manifest(self):
        """Refresh the manifest using the manifest generator script."""
        try:
            self.generate_fresh_manifest()
        except Exception as e:
            raise Exception(f"Failed to refresh manifest: {e}")

    def download_videos(self, num_videos, language_code=None, project_name=None):
        """
        Download a specified number of videos from DBL-sign.

        Args:
            num_videos: Number of videos to download
            language_code: Optional filter by language code
            project_name: Optional filter by project name

        Returns:
            List of dictionaries with video information
        """
        if "languages" not in self.manifest:
            raise ValueError("Invalid manifest structure: 'languages' key not found")

        # Filter projects based on criteria
        filtered_projects = []
        total_languages = len(self.manifest["languages"])
        print(f"Total Languages in Manifest: {total_languages}")
        for i, (lang_code, projects) in enumerate(self.manifest["languages"].items()):
            if language_code and lang_code != language_code:
                continue

            for proj_name, proj_info in projects.items():
                # Only filter by project name if it's explicitly provided
                if project_name and proj_name != project_name:
                    continue

                # Count MP4 files in this project
                mp4_count = sum(
                    1
                    for file_info in proj_info["files"]
                    if file_info["filename"].lower().endswith(".mp4")
                )

                if mp4_count > 0:
                    filtered_projects.append((lang_code, proj_name, proj_info))

        if not filtered_projects:
            filter_criteria = f"language_code={language_code}"
            if project_name:
                filter_criteria += f", project_name={project_name}"
            error_msg = f"No projects found matching the criteria ({filter_criteria})"

            raise ValueError(error_msg)

        # Download videos
        downloaded_videos = []
        videos_downloaded = 0

        print(f"Found {len(filtered_projects)} projects matching the criteria")

        for i, (lang_code, proj_name, proj_info) in enumerate(filtered_projects):
            if videos_downloaded >= num_videos:
                break

            project_dir = os.path.join(self.output_dir, lang_code, proj_name)
            print(f"Project Dir: {project_dir}")
            os.makedirs(project_dir, exist_ok=True)

            # Get metadata for this project
            metadata = {
                "language_code": lang_code,
                "project_name": proj_name,
                "copyright": proj_info.get("rights_holder", ""),
                "license": proj_info.get("license", ""),
                "source": proj_info.get("url", ""),
            }

            # Save metadata
            with open(
                os.path.join(project_dir, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(metadata, f, indent=2)

            # Download MP4 files
            mp4_files = [
                f for f in proj_info["files"] if f["filename"].lower().endswith(".mp4")
            ]
            for j, file_info in enumerate(mp4_files):
                if videos_downloaded >= num_videos:
                    break

                filename = file_info["filename"]
                filepath = os.path.join(project_dir, filename)
                url = file_info["download_url"]

                # Skip if file already exists
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    print(f"File already exists: {filepath}")

                    downloaded_videos.append({"path": filepath, "metadata": metadata})
                    videos_downloaded += 1
                    continue

                # Download the file
                print(f"Downloading {url} to {filepath}")

                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()

                    # Show progress bar
                    downloaded_size = 0
                    block_size = 8192

                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)

                    # Validate MP4

                    try:
                        cap = cv2.VideoCapture(filepath)
                        if not cap.isOpened():
                            error_msg = f"Warning: Could not open downloaded file as video: {filepath}"
                            print(error_msg)

                            continue
                        cap.release()

                    except Exception as e:
                        error_msg = f"Error validating video: {e}"
                        print(error_msg)

                        continue

                    downloaded_videos.append({"path": filepath, "metadata": metadata})
                    videos_downloaded += 1

                except Exception as e:
                    error_msg = f"Error downloading {url}: {e}"
                    print(error_msg)

        print(f"Downloaded {videos_downloaded} videos")
        return downloaded_videos


class SignSegmentationProcessor:
    """Process videos with sign-segmentation tools."""

    def __init__(self, output_dir, parent_dir=None):
        """Initialize the processor with output directory."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Paths to sign-segmentation scripts
        if parent_dir is None:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sign_seg_dir = os.path.join(parent_dir, "sign-segmentation")

        # Verify sign-segmentation directory exists
        if not os.path.exists(self.sign_seg_dir):
            print(
                f"Warning: sign-segmentation directory not found at {self.sign_seg_dir}"
            )
            # Try to find it in the current directory
            alt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "sign-segmentation"
            )
            if os.path.exists(alt_path):
                print(
                    f"Found sign-segmentation directory at alternate location: {alt_path}"
                )
                self.sign_seg_dir = alt_path

        # Verify required script files exist
        required_scripts = {
            "segment_video.py": None,
        }

        for script in required_scripts:
            script_path = os.path.join(self.sign_seg_dir, script)
            if not os.path.exists(script_path):
                print(f"Warning: Required script {script} not found at {script_path}")
            else:
                required_scripts[script] = script_path

        # Import the modules
        try:
            self.segment_video = import_module_from_file(
                "segment_video",
                required_scripts["segment_video.py"]
                or os.path.join(self.sign_seg_dir, "segment_video.py"),
            )
            print("Successfully imported segment_video module")
        except Exception as e:
            print(f"Error importing segment_video module: {str(e)}")
            self.segment_video = None

    def process_video(self, video_path, metadata):
        """
        Process a video with sign-segmentation.

        Args:
            video_path: Path to the video file
            metadata: Metadata dictionary for the video

        Returns:
            List of dictionaries with segment information
        """
        # Verify that the video file exists
        if not video_path or video_path == "path" or not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return []

        print(f"Processing video: {video_path}")
        print(f"Video exists: {os.path.exists(video_path)}")
        print(f"Video size: {os.path.getsize(video_path)} bytes")

        # Verify that the required modules were imported successfully
        if not self.segment_video:
            print("Error: segment_video module not available. Cannot process video.")
            # Try to find segments that might have been created previously
            print(f"Looking for pre-existing segments for {video_path}...")
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Check possible segment locations
            possible_segment_dirs = [
                os.path.join(os.path.dirname(video_path), "..", "segments"),
                os.path.join(os.path.dirname(video_path), "segments"),
                os.path.join(self.output_dir, "segments"),
                self.output_dir,
            ]

            for segments_dir in possible_segment_dirs:
                if os.path.exists(segments_dir):
                    try:
                        segment_files = [
                            f
                            for f in os.listdir(segments_dir)
                            if f.endswith(".mp4") and video_name in f
                        ]
                        if segment_files:
                            print(
                                f"Found {len(segment_files)} pre-existing segments in {segments_dir}"
                            )
                            # Process these segments
                            return self._process_existing_segments(
                                segments_dir, segment_files, video_name, metadata
                            )
                    except Exception as e:
                        print(
                            f"Error checking for segments in {segments_dir}: {str(e)}"
                        )

            print(f"No pre-existing segments found for {video_path}")
            return []

        # Create a unique output directory for this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Created video output directory: {video_output_dir}")

        # Create segments directory
        segments_output_dir = os.path.join(
            os.path.dirname(video_path), "..", "segments"
        )
        os.makedirs(segments_output_dir, exist_ok=True)
        print(f"Created segments directory: {Path(segments_output_dir).resolve()}")

        # Verify segments directory exists
        if not os.path.exists(segments_output_dir):
            print(f"Error: Failed to create segments directory: {segments_output_dir}")

        print(f"Segments will be stored in: {segments_output_dir}")

        # Process the video with sign-segmentation

        print(f"Segmenting video {video_path}")

        # Call the process_video function from segment_video module
        print(f"Calling segment_video.process_video with {video_path}")
        result = self.segment_video.process_video(video_path)
        print(f"Segmentation result: {result}")
        exit()

        # Check if any segments were created
        try:
            segment_files = [
                f
                for f in os.listdir(segments_output_dir)
                if f.endswith(".mp4") and video_name in f
            ]
            print(f"Found {len(segment_files)} segments in {segments_output_dir}")

            # Try alternative locations if no segments found
            if not segment_files:
                alt_dirs = [
                    os.path.join(os.path.dirname(video_path), "segments"),
                    os.path.join(self.output_dir, "segments"),
                    video_output_dir,
                ]
                for alt_dir in alt_dirs:
                    if os.path.exists(alt_dir):
                        try:
                            alt_segment_files = [
                                f
                                for f in os.listdir(alt_dir)
                                if f.endswith(".mp4") and video_name in f
                            ]
                            if alt_segment_files:
                                print(
                                    f"Found {len(alt_segment_files)} segments in alternative directory: {alt_dir}"
                                )
                                segments_output_dir = alt_dir
                                segment_files = alt_segment_files
                                break
                        except Exception as e:
                            print(f"Error checking for segments in {alt_dir}: {str(e)}")
        except Exception as e:
            print(f"Error checking for segments after processing: {str(e)}")
            segment_files = []

        if not segment_files:
            print(f"Warning: No segments were created for {video_path}")
            return []

        # Process the segments
        return self._process_existing_segments(
            segments_output_dir, segment_files, video_name, metadata
        )

    def _process_existing_segments(
        self, segments_dir, segment_files, video_name, metadata
    ):
        """
        Process existing segment files without running segmentation.

        Args:
            segments_dir: Directory containing segment files
            segment_files: List of segment filenames
            video_name: Name of the original video
            metadata: Metadata dictionary for the video

        Returns:
            List of dictionaries with segment information
        """
        print(f"Processing {len(segment_files)} pre-existing segments")
        segments_info = []

        # Filter out files that already have processing suffixes to avoid duplicate processing
        filtered_segment_files = []
        for segment_file in segment_files:
            base_name = os.path.basename(segment_file)
            # Skip files that already have processing suffixes
            if any(
                suffix in base_name
                for suffix in [".original.", ".pose.", ".mask.", ".segmentation."]
            ):
                continue
            filtered_segment_files.append(segment_file)

        print(
            f"Found {len(filtered_segment_files)} segments to process after filtering"
        )

        for i, segment_file in enumerate(filtered_segment_files):
            segment_name = os.path.basename(segment_file).replace(".mp4", "")
            segment_path = os.path.join(segments_dir, f"{segment_name}.mp4")

            # Skip if the segment file doesn't exist
            if not os.path.exists(segment_path):
                logger.log_warning(
                    f"Segment file does not exist: {segment_path}", segment_name
                )
                continue

            original_path = os.path.join(segments_dir, f"{segment_name}.original.mp4")

            # Create a copy of the original segment file if it doesn't exist
            if not os.path.exists(original_path):
                shutil.copy(segment_path, original_path)
                logger.log_info(
                    f"Created copy of original segment file: {original_path}",
                    segment_name,
                )

            # Only add to segment_info if all files were created successfully
            if True:
                segment_info = {
                    "segment_name": segment_name,
                    "original": original_path,
                    "segment_path": original_path,  # Use original path as the main segment path
                    "segment_metadata": {
                        **metadata,
                        "segment_index": i,
                        "segment_count": len(filtered_segment_files),
                        "segment_name": segment_name,
                        "video_name": video_name,
                    },
                }

                segments_info.append(segment_info)
            else:
                missing_files = []

                logger.log_warning(
                    f"Skipping segment due to missing files: {', '.join(missing_files)}",
                    segment_name,
                )

        print(f"Successfully processed {len(segments_info)} pre-existing segments")
        return segments_info


class WebDatasetCreator:
    """Class to handle creating WebDataset format."""

    def __init__(self, output_dir="webdataset"):
        """Initialize the creator with output directory."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def create_webdataset(self, segments_info, shard_size=1000):
        """
        Create WebDataset from segments.

        Args:
            segments_info: List of dictionaries with segment information
            shard_size: Number of segments per WebDataset shard

        Returns:
            List of created WebDataset shard paths
        """
        # Create shards
        shard_paths = []
        os.makedirs(self.output_dir, exist_ok=True)

        # Group segments by shard
        shards = []
        current_shard = []

        for segment_info in segments_info:
            current_shard.append(segment_info)
            if len(current_shard) >= shard_size:
                shards.append(current_shard)
                current_shard = []

        if current_shard:
            shards.append(current_shard)

        # Create each shard
        for i, shard in enumerate(shards):
            shard_path = os.path.join(self.output_dir, f"shard_{i:05d}.tar")
            with tarfile.open(shard_path, "w") as tar:
                for segment_info in shard:
                    segment_name = segment_info["segment_name"]

                    # Check if segment_path exists, if not, use original key
                    if (
                        "segment_path" not in segment_info
                        and "original" in segment_info
                    ):
                        segment_path = segment_info["original"]
                    elif "segment_path" in segment_info:
                        segment_path = segment_info["segment_path"]
                    else:
                        print(
                            f"Warning: No path found for segment {segment_name}. Skipping."
                        )
                        continue

                    # Get the base segment name without any extensions
                    base_segment_name = os.path.splitext(
                        os.path.basename(segment_path)
                    )[0]

                    # Remove any existing subcategory suffixes
                    for suffix in [".original", ".pose", ".mask", ".segmentation"]:
                        if base_segment_name.endswith(suffix):
                            base_segment_name = base_segment_name[: -len(suffix)]

                    # Define the paths for all file types
                    original_path = os.path.join(
                        os.path.dirname(segment_path),
                        f"{base_segment_name}.original.mp4",
                    )
                    pose_path = os.path.join(
                        os.path.dirname(segment_path), f"{base_segment_name}.pose.mp4"
                    )
                    mask_path = os.path.join(
                        os.path.dirname(segment_path), f"{base_segment_name}.mask.mp4"
                    )
                    segmentation_path = os.path.join(
                        os.path.dirname(segment_path),
                        f"{base_segment_name}.segmentation.mp4",
                    )

                    # Create a sample with all available files
                    sample = {}

                    # Add the original video
                    if (
                        os.path.exists(original_path)
                        and os.path.getsize(original_path) > 0
                    ):
                        sample["mp4"] = original_path
                    elif (
                        os.path.exists(segment_path)
                        and os.path.getsize(segment_path) > 0
                    ):
                        sample["mp4"] = segment_path
                    else:
                        print(
                            f"Warning: Original video file not found for {segment_name}. Skipping."
                        )
                        continue

                    # Add the pose video if available
                    if os.path.exists(pose_path) and os.path.getsize(pose_path) > 0:
                        sample["pose.mp4"] = pose_path
                        segment_info["segment_metadata"]["pose"] = (
                            f"{base_segment_name}.pose.mp4"
                        )
                    else:
                        segment_info["segment_metadata"]["pose"] = None

                    # Add the mask video if available
                    if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                        sample["mask.mp4"] = mask_path
                        segment_info["segment_metadata"]["mask"] = (
                            f"{base_segment_name}.mask.mp4"
                        )
                    else:
                        segment_info["segment_metadata"]["mask"] = None

                    # Add the segmentation video if available
                    if (
                        os.path.exists(segmentation_path)
                        and os.path.getsize(segmentation_path) > 0
                    ):
                        sample["segmentation.mp4"] = segmentation_path
                        segment_info["segment_metadata"]["segmentation"] = (
                            f"{base_segment_name}.segmentation.mp4"
                        )
                    else:
                        segment_info["segment_metadata"]["segmentation"] = None

                    # Add the metadata
                    sample["json"] = json.dumps(segment_info["segment_metadata"])

                    # Add the sample to the shard
                    for key, path in sample.items():
                        if key == "json":
                            # Add the JSON metadata directly
                            info = tarfile.TarInfo(f"{segment_name}_{i}.{key}")
                            json_bytes = path.encode("utf-8")
                            info.size = len(json_bytes)
                            tar.addfile(info, io.BytesIO(json_bytes))
                        else:
                            # Add the file using a binary read to avoid encoding issues
                            try:
                                # Get file info
                                file_size = os.path.getsize(path)
                                if file_size == 0:
                                    print(
                                        f"Warning: File {path} has zero size. Skipping."
                                    )
                                    continue

                                # Create tar info
                                info = tarfile.TarInfo(f"{segment_name}_{i}.{key}")
                                info.size = file_size

                                # Add file with binary mode to avoid encoding issues
                                with open(path, "rb") as file_data:
                                    tar.addfile(info, file_data)
                            except Exception as e:
                                print(f"Error adding file {path} to tar: {e}")

            shard_paths.append(shard_path)

        return shard_paths


def process_without_gui(args):
    """Process videos without GUI updates."""
    # Initialize directories
    os.makedirs(args.output_dir, exist_ok=True)
    downloads_dir = os.path.join(args.output_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    processed_dir = os.path.join(args.output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Step 1: Download videos
    print("=== Step 1: Downloading videos from DBL-sign ===")
    downloader = DBLSignDownloader(downloads_dir)
    video_info_list = downloader.download_videos(
        args.num_videos, args.language_code, args.project_name
    )
    print(video_info_list)
    print(json.dumps(video_info_list, indent=4))

    # Step 2: Process videos with sign-segmentation
    print("=== Step 2: Processing videos with sign-segmentation ===")
    processor = SignSegmentationProcessor(processed_dir)

    all_segments = []
    random.shuffle(video_info_list)
    for i, video_info in enumerate(video_info_list):
        # Extract video path and metadata from the dictionary
        video_path = video_info["path"]
        metadata = video_info["metadata"]

        # Verify video path is valid
        if not video_path or not os.path.exists(video_path):
            print(f"Skipping invalid video path: {video_path}")
            continue

        segments = processor.process_video(video_path, metadata)
        all_segments.extend(segments)
    print(all_segments)
    exit()
    # Step 3: Create WebDataset
    print("=== Step 3: Creating WebDataset ===")
    webdataset_dir = os.path.join(args.output_dir, "webdataset")
    os.makedirs(webdataset_dir, exist_ok=True)

    # Create WebDataset
    creator = WebDatasetCreator(webdataset_dir)
    creator.create_webdataset(all_segments, shard_size=args.shard_size)

    # Step 4: Create manifest
    print("=== Step 4: Creating manifest ===")
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    # Create manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"segments": all_segments}, f, indent=2)

    print(f"Processing complete! Manifest saved to {manifest_path}")


def main():
    """Main function to run the entire process."""
    parser = argparse.ArgumentParser(
        description="Prepare sign language videos for HuggingFace datasets"
    )
    parser.add_argument(
        "--num-videos", type=int, default=10, help="Number of videos to download"
    )
    parser.add_argument("--language-code", type=str, help="Filter by language code")
    parser.add_argument("--project-name", type=str, help="Filter by project name")

    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of segments per WebDataset shard",
    )
    args = parser.parse_args()

    process_without_gui(args)


if __name__ == "__main__":
    main()
# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/prepare_webdataset.py --output-dir . --language-code esl
