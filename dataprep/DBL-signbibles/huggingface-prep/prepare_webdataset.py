#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
1. Downloads videos from DBL-sign
2. Processes them with sign-segmentation
3. Packages everything into WebDataset format
"""

import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import langcodes
import pandas as pd
import requests
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
manifest_generator_spec = importlib.util.spec_from_file_location("DBL_manifest_generator", manifest_generator_path)
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
            print(f"Error executing module {module_name} from {file_path}: {e!s}")
            return None

        return module
    except Exception as e:
        print(f"Error importing module {module_name} from {file_path}: {e!s}")
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
        with open(self.manifest_path, encoding="utf-8") as f:
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
        with open(self.manifest_path, encoding="utf-8") as f:
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
                mp4_count = sum(1 for file_info in proj_info["files"] if file_info["filename"].lower().endswith(".mp4"))

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
            with open(os.path.join(project_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Download MP4 files
            mp4_files = [f for f in proj_info["files"] if f["filename"].lower().endswith(".mp4")]
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
            print(f"Warning: sign-segmentation directory not found at {self.sign_seg_dir}")
            # Try to find it in the current directory
            alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign-segmentation")
            if os.path.exists(alt_path):
                print(f"Found sign-segmentation directory at alternate location: {alt_path}")
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
                required_scripts["segment_video.py"] or os.path.join(self.sign_seg_dir, "segment_video.py"),
            )
            print("Successfully imported segment_video module")
        except Exception as e:
            print(f"Error importing segment_video module: {e!s}")
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

        segments_dir = video_path.parent / "segments"

        print(f"Processing video: {video_path}")
        print(f"Video exists: {os.path.exists(video_path)}")
        print(f"Video size: {os.path.getsize(video_path)} bytes")

        # Verify that the required modules were imported successfully
        if not self.segment_video:
            print("Error: segment_video module not available. Cannot process video.")
            # Try to find segments that might have been created previously
            print(f"Looking for pre-existing segments for {video_path}...")
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            segment_files = [f for f in os.listdir(segments_dir) if f.endswith(".mp4") and video_name in f]
            if segment_files:
                print(f"Found {len(segment_files)} pre-existing segments in {segments_dir}")
                # Process these segments
                return self._process_existing_segments(segments_dir, segment_files, video_name, metadata)

            print(f"No pre-existing segments found for {video_path}")
            return []

        # Create a unique output directory for this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Created video output directory: {video_output_dir}")

        # Create segments directory
        os.makedirs(segments_dir, exist_ok=True)
        print(f"Created segments directory: {Path(segments_dir).resolve()}")

        # Verify segments directory exists
        if not os.path.exists(segments_dir):
            print(f"Error: Failed to create segments directory: {segments_dir}")

        print(f"Segments will be stored in: {segments_dir}")

        # Process the video with sign-segmentation

        print(f"Segmenting video {video_path}")

        # Call the process_video function from segment_video module
        print(f"Calling segment_video.process_video with {video_path}")
        self.segment_video.process_video(video_path)

        # Check if any segments were created

        segment_files = [f for f in os.listdir(segments_dir) if f.endswith(".mp4") and video_name in f]
        print(f"Found {len(segment_files)} segments in {segments_dir}")

        if not segment_files:
            print(f"Warning: No segments were created for {video_path}")
            return []

        # Process the segments
        return self._process_existing_segments(segments_dir, segment_files, video_name, metadata)

    def _process_existing_segments(self, segments_dir, segment_files, video_name, metadata):
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
            if any(suffix in base_name for suffix in [".original.", ".pose.", ".mask.", ".segmentation."]):
                continue
            filtered_segment_files.append(segment_file)

        print(f"Found {len(filtered_segment_files)} segments to process after filtering")

        for i, segment_file in enumerate(filtered_segment_files):
            segment_name = os.path.basename(segment_file).replace(".mp4", "")
            segment_path = os.path.join(segments_dir, f"{segment_name}.mp4")

            # Skip if the segment file doesn't exist
            if not os.path.exists(segment_path):
                logger.log_warning(f"Segment file does not exist: {segment_path}", segment_name)
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

    def __init__(self, output_dir: str = "webdataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_webdataset(self, segments_info: list[dict[str, Any]], shard_size: int = 1000) -> list[str]:
        shards = self._split_into_shards(segments_info, shard_size)
        return self._write_shards(shards)

    def _split_into_shards(self, segments_info: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
        return [segments_info[i : i + shard_size] for i in range(0, len(segments_info), shard_size)]

    def _write_shards(self, shards: list[list[dict[str, Any]]]) -> list[str]:
        shard_paths = []

        for shard_index, shard in enumerate(shards):
            shard_path = self.output_dir / f"shard_{shard_index:05d}.tar"
            with tarfile.open(shard_path, "w") as tar:
                for segment_info in shard:
                    sample = self._build_sample(segment_info, shard_index)
                    if not sample:
                        continue
                    self._add_sample_to_tar(tar, sample, segment_info["segment_name"])
            shard_paths.append(str(shard_path))

        return shard_paths

    def _build_sample(self, segment_info: dict[str, Any], shard_index: int) -> dict[str, str | Path]:
        print(json.dumps(segment_info, indent=2))
        sample = {}

        sample["json"] = json.dumps(segment_info["segment_metadata"])
        sample["mp4"] = segment_info["segment_path"]
        print(segment_info)
        for ext, path in segment_info["files_to_add"]:
            sample[ext] = path

        # sample["pose-animation.mp4"] = segment_info["segment_metadata"]["pose"]["animation"]
        # sample["pose-dwpose.npz"] = segment_info["segment_metadata"]["pose"]["dwpose"]
        # sample["pose"] = segment_info["segment_metadata"]["pose"]["mediapipe"]
        # sample["segment_name"] = segment_info["segment_name"]

        return sample

    def _add_sample_to_tar(
        self,
        tar: tarfile.TarFile,
        sample: dict[str, str | Path],
        segment_name: str,
    ):
        for ext, content in sample.items():
            filename = f"{segment_name}.{ext}"
            try:
                if ext == "json":
                    encoded = content.encode("utf-8")  # type: ignore
                    info = tarfile.TarInfo(filename)
                    info.size = len(encoded)
                    tar.addfile(info, io.BytesIO(encoded))
                else:
                    content = Path(content)  # type: ignore
                    size = content.stat().st_size
                    if size == 0:
                        print(f"Warning: File {content} has zero size. Skipping.")
                        continue
                    info = tarfile.TarInfo(filename)
                    info.size = size
                    with content.open("rb") as f:
                        tar.addfile(info, f)
            except Exception as e:
                print(f"Error adding {filename} to tar: {e}")


def parse_metadata_to_bible_ref(xml_path: Path, filename):
    """
    Parse a metadata.xml and return bible ref associated, or empty string
    e.g. "CBT-001-esl-3_Bible_God Creates Everything.mp4" -> "GEN 1:1-31,2:1-3"
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # dbl_id = root.attrib.get("id", None)
        file_to_passage = {}

        for pub in root.findall("./publications/publication"):
            for division in pub.findall("./structure/division"):
                passage = division.attrib.get("role", "").strip()
                for content in division.findall("content"):
                    src = content.attrib.get("src", "").strip()
                    filename = Path(src).name
                    file_to_passage[filename] = passage

        return file_to_passage.get(filename, "")

    except ET.ParseError as e:
        print(f"[ERROR] Could not parse {xml_path}: {e}")
        return None, {}


def search_ebible_translations(
    language_code: str, translation_id: str, ebible_translations_df: pd.DataFrame
) -> dict | None:
    result = ebible_translations_df[
        (ebible_translations_df["languageCode"] == language_code)
        & (ebible_translations_df["translationId"] == translation_id)
    ]
    if result.empty:
        print(f"No results found for ({language_code}, {translation_id})")
        return None
    elif len(result) > 1:
        print(f"Warning: Multiple results found for ({language_code}, {translation_id}), returning the first.")

    return result.iloc[0].to_dict()


def load_vref_map(vref_path: str) -> dict[str, int]:
    with open(vref_path, encoding="utf-8") as f:
        return {line.strip(): idx for idx, line in enumerate(f) if line.strip()}


def load_bible_lines(bible_path: str) -> list[str]:
    with open(bible_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def parse_citation_string(citation: str, vref_map: dict[str, int]) -> list[int]:
    all_indices = []
    current_book = None
    current_chapter = None
    tokens = re.split(r";\s*", citation.strip())

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)\s+(\d+)", token)
        if m:
            book, chapter = m.groups()
            current_book = book
            current_chapter = chapter
            prefix = f"{book} {chapter}:"
            matches = [i for ref, i in vref_map.items() if ref.startswith(prefix)]
            all_indices.extend(matches)
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)", token)
        if m:
            book = m.group(1)
            current_book = book
            matches = [i for ref, i in vref_map.items() if ref.startswith(f"{book} ")]
            all_indices.extend(matches)
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+:\d+)\s*-\s*([1-3]?[A-Z]+)?\s*(\d+:\d+)", token)
        if m:
            book1, start, book2, end = m.groups()
            if book1:
                current_book = book1
            book2 = book2 or current_book
            if not (current_book and book2):
                continue
            start_ref = f"{current_book} {start}"
            end_ref = f"{book2} {end}"
            if start_ref in vref_map and end_ref in vref_map:
                i1, i2 = vref_map[start_ref], vref_map[end_ref]
                if i1 <= i2:
                    all_indices.extend(range(i1, i2 + 1))
            continue

        parts = token.split(",")
        for part in parts:
            part = part.strip()
            m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+):(\d+(?:-\d+)?)", part)
            if m:
                maybe_book, ch, verse_range = m.groups()
                if maybe_book:
                    current_book = maybe_book
                current_chapter = ch
                if not current_book:
                    continue

                if "-" in verse_range:
                    start_v, end_v = map(int, verse_range.split("-"))
                    verse_numbers = range(start_v, end_v + 1)
                else:
                    verse_numbers = [int(verse_range)]

                for v in verse_numbers:
                    ref = f"{current_book} {current_chapter}:{v}"
                    if ref in vref_map:
                        all_indices.append(vref_map[ref])
    return sorted(set(all_indices))


def citation_to_text_and_vrefs(citation: str, vref_map, bible_verses):
    vrefs = parse_citation_string(citation, vref_map)

    verses = [bible_verses[i] for i in vrefs if 0 <= i < len(bible_verses)]

    bible_text = "".join(verses)
    return bible_text, vrefs


def process_without_gui(args):
    """Process videos without GUI updates using pathlib for filesystem operations."""
    # Initialize paths
    output_dir = Path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    processed_dir = output_dir / "processed"
    webdataset_dir = output_dir / "webdataset"
    manifest_path = output_dir / "manifest.json"

    # eBible Corpus
    ebible_corpus_path = args.ebible_corpus_path
    vref_text_path = ebible_corpus_path / "metadata" / "vref.txt"
    ebible_translations_csv_path = ebible_corpus_path / "metadata" / "translations.csv"
    ebible_text_path = ebible_corpus_path / "corpus" / f"{args.ebible_version}.txt"
    if not ebible_corpus_path.is_dir():
        raise ValueError(
            f"eBible Corpus is not at {ebible_corpus_path}. Please clone it there or pass --ebible-corpus-path"
        )

    vref_map = load_vref_map(vref_text_path)
    bible_verses = load_bible_lines(ebible_text_path)
    print(f"Loading vref-to-index map from {vref_text_path}: {len(vref_map)} keys")
    print(f"Loading {args.ebible_version} verses from {ebible_text_path}: {len(bible_verses)} loaded")
    ebible_translations_df = pd.read_csv(ebible_translations_csv_path)
    print(f"Loading translations info from {ebible_translations_csv_path}: {len(ebible_translations_df)} translations")

    # Assert uniqueness of (languageCode, translationId)
    # duplicates = ebible_translations_df.duplicated(subset=["languageCode", "translationId"], keep=False)
    # assert not duplicates.any(), f"Duplicate rows found:\n{ebible_translations_df[duplicates]}"
    language_code, translation_id = args.ebible_version.split("-")
    ebible_version_metadata = search_ebible_translations(language_code, translation_id, ebible_translations_df)
    print(f"Metadata for {args.ebible_version}: {ebible_version_metadata}")

    # Create directories
    downloads_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download videos
    print("=== Step 1: Downloading videos from DBL-sign ===")
    downloader = DBLSignDownloader(downloads_dir)
    video_info_list = downloader.download_videos(args.num_videos, args.language_code, args.project_name)
    print(video_info_list)
    print(json.dumps(video_info_list, indent=4))

    # ---------------------------------------
    # Parse verse data from metadata.xml
    video_info_list_with_bible_refs = []
    for video_info in video_info_list:
        video_info["filename"] = Path(video_info["path"]).name
        meta_xml = Path(video_info["path"]).parent / "metadata.xml"
        print(f"Metadata for {video_info['path']}: {meta_xml}")

        video_info["transcripts"] = []

        video_info["bible-ref"] = parse_metadata_to_bible_ref(meta_xml, video_info)
        # print(file_to_passage)
        transcript = {}
        bible_text, vrefs = citation_to_text_and_vrefs(
            video_info["bible-ref"], vref_map=vref_map, bible_verses=bible_verses
        )
        video_info["biblenlp-vref"] = vrefs
        transcript["bible_text"] = bible_text
        # print(ebible_version_metadata)
        # {'languageCode': 'eng', 'translationId': 'engbsb', 'languageName': 'English', 'languageNameInEnglish': 'English', 'dialect': nan, 'homeDomain': 'ebible.org', 'title': 'Berean Standard Bible', 'description': 'The Holy Bible in English: Berean Standard Bible', 'Redistributable': True, 'Copyright': 'public domain', 'UpdateDate': '2024-07-13', 'publicationURL': 'http://ebible.org/engbsb/', 'OTbooks': 39, 'OTchapters': 929, 'OTverses': 23145, 'NTbooks': 27, 'NTchapters': 260, 'NTverses': 7941, 'DCbooks': 0, 'DCchapters': 0, 'DCverses': 0, 'FCBHID': 'ENGBSB', 'Certified': True, 'inScript': 'http://eBible.org/study/?v1=GN1_1&w1=bible&t1=local%3A', 'swordName': 'engbsb2020eb', 'rodCode': nan, 'textDirection': 'ltr', 'downloadable': True, 'font': 'DejaVu Serif', 'shortTitle': 'English Berean Standard Bible', 'PODISBN': nan, 'script': 'Latin', 'sourceDate': '2024-07-13'}
        transcript["language"] = {
            "name": ebible_version_metadata["languageName"],
            "ISO639-3": ebible_version_metadata["languageCode"],
            "BCP-47": langcodes.standardize_tag(ebible_version_metadata["languageCode"]),
            "license": ebible_version_metadata["Copyright"],
            "source": ebible_version_metadata["publicationURL"],
            # "BCP-47": "en-US",
        }
        # transcript["source"] = "Berean Standard Bible"
        transcript["source"] = ebible_version_metadata["title"]
        video_info["transcripts"].append(transcript)
        print(video_info)
        # ebible_version_metadata

        video_info_list_with_bible_refs.append(video_info)

    video_info_list = video_info_list_with_bible_refs

    # Step 2: Process videos with sign-segmentation
    print("=== Step 2: Processing videos with sign-segmentation ===")
    if args.auto_segment:
        processor = SignSegmentationProcessor(processed_dir)

        # only keep the "Bible" or "Passage" videos
        # won't work for eso
        print("Filtering for 'Bible' or 'Passage' videos")
        filtered_video_info_list = [
            video_info
            for video_info in video_info_list
            if "Passage" in video_info["path"] or "Bible" in video_info["path"]
        ]
        print(f"{len(filtered_video_info_list)} videos left")
        if len(filtered_video_info_list) == 0:
            print("Project has no matching videos with those keywords keeping the whole list")
        else:
            video_info_list = filtered_video_info_list

        all_segments = []
        for i, video_info in enumerate(video_info_list[:2]):
            video_path = Path(video_info["path"])
            metadata = video_info["metadata"]

            if not video_path.exists():
                print(f"Skipping invalid video path {i}: {video_path}")
                continue

            segments = processor.process_video(video_path, metadata)
            all_segments.extend(segments)

        print(f"{len(all_segments)} segments")
    else:
        print("--auto-segment not set, returning original files")
        all_segments = []
        for video_info in video_info_list:
            video_path = Path(video_info["path"])
            del video_info["path"]
            video_info["pose"] = {}

            files_to_add = []
            pose_animation_path = video_path.with_suffix(".pose-animation.mp4")
            if pose_animation_path.is_file():
                video_info["pose"]["animation"] = str(pose_animation_path.name)
                files_to_add.append(("pose-animation.mp4", str(pose_animation_path)))

            dw_pose_path = video_path.with_suffix(".pose-dwpose.npz")
            if dw_pose_path.is_file():
                video_info["pose"]["dwpose"] = str(dw_pose_path.name)
                files_to_add.append(("pose-dwpose.npz", str(dw_pose_path)))

            mediapipe_path = video_path.with_suffix(".pose")
            if mediapipe_path.is_file():
                video_info["pose"]["mediapipe"] = str(mediapipe_path.name)
                files_to_add.append(("pose", str(mediapipe_path)))

            print(video_info)

            segment_info = {
                "segment_name": video_path.stem,
                "original": str(video_path),
                "segment_path": str(video_path),  # Use original path as the main segment path
                "segment_metadata": {
                    **video_info,
                    "segment_index": 0,
                    "segment_count": 1,
                    "segment_name": video_path.stem,
                    "video_name": video_path.stem,
                },
                "files_to_add": files_to_add,
            }

            print(segment_info)
            all_segments.append(segment_info)

    # Step #: Create WebDatasetF
    print("=== Step 3: Creating WebDataset ===")
    webdataset_dir.mkdir(parents=True, exist_ok=True)
    print(webdataset_dir.resolve())

    creator = WebDatasetCreator(webdataset_dir)
    creator.create_webdataset(all_segments, shard_size=args.shard_size)

    # Step 4: Create manifest
    print("=== Step 4: Creating manifest ===")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"segments": all_segments}, f, indent=2)

    print(f"Processing complete! Manifest saved to {manifest_path}")


def main():
    """Main function to run the entire process."""
    parser = argparse.ArgumentParser(description="Prepare sign language videos for HuggingFace datasets")
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to download")
    parser.add_argument("--language-code", type=str, help="Filter by language code")
    parser.add_argument("--project-name", type=str, help="Filter by project name")
    parser.add_argument(
        "--auto-segment",
        type=str,
        help="Use autosegmentation from sign-language-processing",
    )
    parser.add_argument(
        "--ebible-corpus-path",
        type=Path,
        help="Root directory of the eBible corpus. https://github.com/BibleNLP/ebible",
        default=Path(parent_dir) / "ebible/",
    )
    parser.add_argument(
        "--ebible-version",
        type=str,
        help="Which Bible from the eBible corpus to use for verse texts",
        default="eng-engbsb",
    )

    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
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
