#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
1. Downloads videos from DBL-sign
2. Processes them with sign-segmentation
3. Packages everything into WebDataset format
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path
from tqdm import tqdm
import webdataset as wds
import cv2
import numpy as np
import requests
import boto3
import time
import importlib.util
import threading
import tarfile
import io
import glob
from processing_logger import ProcessingLogger, LogLevel

# Initialize the logger with the correct path
logger = ProcessingLogger(log_file_path="../output/run_log.txt")
# Clear the log at the start of the process
logger.clear_log()

# Add command line info to the log
logger.log_info(f"Command: {' '.join(sys.argv)}")

# Add gui import if available
gui = False
try:
    from progress_gui import start_gui, update_gui
    gui = True
except ImportError:
    # Fallback if progress_gui is not available
    print("Warning: progress_gui module not found. Using console output instead.")
    
    def start_gui(*args, **kwargs):
        return None
        
    def update_gui(*args, **kwargs):
        pass

try:
    from gui_utils import update_progress
    gui = True
except ImportError:
    # Fallback if gui_utils is not available
    def update_progress(*args, **kwargs):
        # Extract the message if available and print it
        if len(args) > 1 and args[1]:
            print(args[1])
        elif kwargs.get('message'):
            print(kwargs.get('message'))

# Global GUI variables
root = None

# Add parent directory to path for importing from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
dbl_sign_dir = os.path.join(parent_dir, "DBL-sign")
sign_seg_dir = os.path.join(parent_dir, "sign-segmentation")
sys.path.append(dbl_sign_dir)
sys.path.append(sign_seg_dir)

# Import DBL-sign utilities
from dbl_utils import DownloadLog
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

# Try to import the refine_mask_v2 module
refine_mask_v2_path = os.path.join(sign_seg_dir, "refine_mask_v2.py")
refine_mask_v2 = import_module_from_file("refine_mask_v2", refine_mask_v2_path)
if refine_mask_v2:
    print("Successfully imported refine_mask_v2 module")
else:
    print("Warning: Failed to import refine_mask_v2 module. Advanced segmentation will not be available.")

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
        
        # Update GUI progress if available
        if gui:
            update_progress("Generating manifest", "Preparing to generate fresh manifest...", 10, 0)
        
        # Response data containing the list of sign language projects (hardcoded in the original script)
        response_data = {"aaData": [
            ["245cca6ac5984130", "Sri Lankan Sign Language", "sqs", "Sri Lanka", "Chronological Bible Translation in Sri Lankan Sign Language", "DOOR International"],
            ["055195093e2347d0", "Burundian Sign Language", "lsb", "Burundi", "Chronological Bible Translation in Burundian Sign Language", "DOOR International"],
            ["0a3ea85ee1e34a2d", "Nepali Sign Language", "nsp", "Nepal", "Chronological Bible Translation in Nepali Sign Language", "DOOR International"],
            ["1fcac35962494d40", "Ugandan Sign Language", "ugn", "Uganda", "Chronological Bible Translation in Ugandan Sign Language", "DOOR International"],
            ["a63aef8004db401b", "Ethiopian Sign Language", "eth", "Ethiopia", "Chronological Bible Translation in Ethiopian Sign Language", "DOOR International"],
            ["56c922b7b5a44a47", "Kerala Sign Language", "mis", "India", "Chronological Bible Translation in Kerala Sign Language", "DOOR International"],
            ["65c350c1cf9c42e4", "Nigerian Sign Language", "nsi", "Federal Republic Nigeria", "Chronological Bible Translation in Nigerian Sign Language", "DOOR International"],
            ["6ad9cf57ab084a3b", "Estonian Sign Language", "eso", "Estonia", "The Bible in Estonian Sign Language", "Deaf Bible Society"],
            ["9e9e20d036fa4e91", "Tanzanian Sign Language", "tza", "Tanzania", "Chronological Bible Translation in Tanzanian Sign Language", "DOOR International"],
            ["6c1ffbf874d14ee1", "Andhra Pradesh Sign Language", "mis", "India", "Chronological Bible Translation in Andhra Pradesh Sign Language", "DOOR International"],
            ["2d6ac5c8b4614955", "Bulgarian Sign Language", "bqn", "Bulgaria", "Chronological Bible Translation in Bulgarian Sign Language", "DOOR International"],
            ["1bacaede20da4494", "American Sign Language", "ase", "United States of America", "Chronological Bible Translation in American Sign Language (119 Introductions and Passages expanded with More Information)", "Deaf Harbor "],
            ["d2027facd4cc4c2a", "American Sign Language", "ase", "United States of America", "Chronological Bible Translation in American Sign Language (119 Introductions and Passages)", "Deaf Harbor "],
            ["6543fec2ced7421d", "South Sudanese Sign Language", "mis", "Republic of South Sudan", "Chronological Bible Translation in South Sudanese Sign Language", "DOOR International"],
            ["a28def50f139432a", "Kenyan Sign Language", "xki", "Kenya, Republic of", "Chronological Bible Translation in Kenyan Sign Language", "DOOR International"],
            ["c4b68657ce9b48ad", "Ghanaian Sign Language", "gse", "Ghana", "Chronological Bible Translation in Ghanaian Sign Language", "DOOR International"],
            ["995240c9d7e8453e", "Indian (Delhi) Sign Language", "ins", "India", "Chronological Bible Translation in Indian (Delhi)  Sign Language", "DOOR International"],
            ["6d5944a5ceb944c0", "Russian Sign Language", "rsl", "Russian Federation", "Chronological Bible Translation in Russian Sign Language", "DOOR International"],
            ["ec8517dba29d4d93", "Egyptian Sign Language", "esl", "Egypt", "Chronological Bible Translation in Egyptian Sign Language", "DOOR International"],
            ["c0b48facec324e4b", "Mozambican Sign Language", "mzy", "Republic of Mozambique", "Chronological Bible Translation in Mozambican Sign Language", "DOOR International"],
            ["b963267b41cc443c", "West Bengal Sign Language", "mis", "India", "Chronological Bible Translation in West Bengal Sign Language", "DOOR International"]
        ]}
        
        # Update GUI progress if available
        if gui:
            update_progress("Generating manifest", "Creating manifest structure...", 10, 30)
        
        # Create manifest using the imported function
        manifest = manifest_generator.create_manifest(response_data)
        
        # Update GUI progress if available
        if gui:
            update_progress("Generating manifest", "Saving manifest to disk...", 10, 70)
        
        # Save manifest using the imported function
        manifest_generator.save_manifest(manifest, self.manifest_path)
        
        # Update GUI progress if available
        if gui:
            update_progress("Generating manifest", "Loading manifest data...", 10, 90)
        
        print("Fresh manifest generated successfully.")
        
        # Load the newly generated manifest
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        # Update GUI progress if available
        if gui:
            update_progress("Generating manifest", "Manifest generation complete", 10, 100)
            
        return self.manifest
    
    def load_manifest(self):
        """Load the manifest file."""
        if not os.path.exists(self.manifest_path):
            if gui:
                update_progress("Generating manifest", "Manifest file not found. Generating a fresh one...", 10, 0)
            print("Manifest file not found. Generating a fresh one...")
            return self.generate_fresh_manifest()
        
        # Check if manifest is older than 24 hours
        manifest_age = time.time() - os.path.getmtime(self.manifest_path)
        if manifest_age > 86400:  # 24 hours in seconds
            if gui:
                update_progress("Generating manifest", "Manifest is older than 24 hours. Generating a fresh one...", 10, 0)
            print("Manifest is older than 24 hours. Generating a fresh one...")
            return self.generate_fresh_manifest()
        
        if gui:
            update_progress("Loading manifest", "Reading existing manifest file...", 10, 50)
        print("Loading existing manifest...")
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        if gui:
            update_progress("Loading manifest", "Manifest loaded successfully", 10, 100)
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
        
        # Update GUI progress if available
        if gui:
            update_progress("Filtering projects", "Finding projects that match criteria...", 10, 0)
        
        # Filter projects based on criteria
        filtered_projects = []
        total_languages = len(self.manifest["languages"])
        for i, (lang_code, projects) in enumerate(self.manifest["languages"].items()):
            # Update GUI progress during filtering
            if gui:
                progress = (i / total_languages) * 50
                update_progress("Filtering projects", f"Checking language: {lang_code}", 10, progress)
                
            if language_code and lang_code != language_code:
                continue
                
            for proj_name, proj_info in projects.items():
                # Only filter by project name if it's explicitly provided
                if project_name and proj_name != project_name:
                    continue
                    
                # Count MP4 files in this project
                mp4_count = sum(1 for file_info in proj_info["files"] 
                               if file_info['filename'].lower().endswith('.mp4'))
                
                if mp4_count > 0:
                    filtered_projects.append((lang_code, proj_name, proj_info))
        
        if not filtered_projects:
            filter_criteria = f"language_code={language_code}"
            if project_name:
                filter_criteria += f", project_name={project_name}"
            error_msg = f"No projects found matching the criteria ({filter_criteria})"
            if gui:
                update_progress("Error", error_msg, 10, 100)
            raise ValueError(error_msg)
        
        # Update GUI progress if available
        if gui:
            update_progress("Projects found", f"Found {len(filtered_projects)} projects matching the criteria", 10, 100)
        
        # Download videos
        downloaded_videos = []
        videos_downloaded = 0
        
        print(f"Found {len(filtered_projects)} projects matching the criteria")
        
        for i, (lang_code, proj_name, proj_info) in enumerate(filtered_projects):
            if videos_downloaded >= num_videos:
                break
                
            # Update GUI progress for project preparation
            if gui:
                update_progress("Preparing project", f"Setting up project {i+1}/{len(filtered_projects)}: {proj_name}", 20, (i / len(filtered_projects)) * 100)
            
            project_dir = os.path.join(self.output_dir, "downloads", lang_code, proj_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Get metadata for this project
            metadata = {
                "language_code": lang_code,
                "project_name": proj_name,
                "copyright": proj_info.get("rights_holder", ""),
                "license": proj_info.get("license", ""),
                "source": proj_info.get("url", "")
            }
            
            # Save metadata
            with open(os.path.join(project_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Download MP4 files
            mp4_files = [f for f in proj_info["files"] if f['filename'].lower().endswith('.mp4')]
            for j, file_info in enumerate(mp4_files):
                if videos_downloaded >= num_videos:
                    break
                    
                filename = file_info['filename']
                filepath = os.path.join(project_dir, filename)
                url = file_info['download_url']
                
                # Update GUI progress for file processing
                if gui:
                    file_progress = (j / len(mp4_files)) * 100
                    update_progress("Processing files", f"Processing file {j+1}/{len(mp4_files)}: {filename}", 20, file_progress)
                
                # Skip if file already exists
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    print(f"File already exists: {filepath}")
                    if gui:
                        update_progress(None, f"File already exists: {filename}", None, 100)
                    downloaded_videos.append({
                        "path": filepath,
                        "metadata": metadata
                    })
                    videos_downloaded += 1
                    continue
                
                # Download the file
                print(f"Downloading {url} to {filepath}")
                
                try:
                    # Update progress if GUI is active
                    if gui:
                        update_progress("Downloading video", f"Downloading {filename}", 30, 0)
                    
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Show progress bar
                    file_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    block_size = 8192
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                
                                # Update download progress if GUI is active
                                if gui and file_size > 0:
                                    download_progress = (downloaded_size / file_size) * 100
                                    update_progress("Downloading video", f"Downloading {filename}: {download_progress:.1f}%", 30, download_progress)
                    
                    # Validate MP4
                    if gui:
                        update_progress("Validating video", f"Validating {filename}", 30, 0)
                    
                    try:
                        cap = cv2.VideoCapture(filepath)
                        if not cap.isOpened():
                            error_msg = f"Warning: Could not open downloaded file as video: {filepath}"
                            print(error_msg)
                            if gui:
                                update_progress("Error", error_msg, 30, 100)
                            continue
                        cap.release()
                        if gui:
                            update_progress("Validating video", f"Successfully validated {filename}", 30, 100)
                    except Exception as e:
                        error_msg = f"Error validating video: {e}"
                        print(error_msg)
                        if gui:
                            update_progress("Error", error_msg, 30, 100)
                        continue
                    
                    downloaded_videos.append({
                        "path": filepath,
                        "metadata": metadata
                    })
                    videos_downloaded += 1
                    
                except Exception as e:
                    error_msg = f"Error downloading {url}: {e}"
                    print(error_msg)
                    # If GUI is active, log the error
                    if gui:
                        update_progress("Error", error_msg, 30, 100)
        
        if gui:
            update_progress("Download complete", f"Downloaded {videos_downloaded} videos", 30, 100)
        
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
            "video_to_pose_mask.py": None,
            "refine_mask_v2.py": None
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
                required_scripts["segment_video.py"] or os.path.join(self.sign_seg_dir, "segment_video.py")
            )
            print(f"Successfully imported segment_video module")
        except Exception as e:
            print(f"Error importing segment_video module: {str(e)}")
            self.segment_video = None
        
        try:
            self.video_to_pose_mask = import_module_from_file(
                "video_to_pose_mask", 
                required_scripts["video_to_pose_mask.py"] or os.path.join(self.sign_seg_dir, "video_to_pose_mask.py")
            )
            print(f"Successfully imported video_to_pose_mask module")
        except Exception as e:
            print(f"Error importing video_to_pose_mask module: {str(e)}")
            self.video_to_pose_mask = None
        
        # Store reference to the refine_mask_v2 module
        self.refine_mask_v2 = refine_mask_v2
    
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
            print(f"Error: segment_video module not available. Cannot process video.")
            # Try to find segments that might have been created previously
            print(f"Looking for pre-existing segments for {video_path}...")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Check possible segment locations
            possible_segment_dirs = [
                os.path.join(os.path.dirname(video_path), '..', 'segments'),
                os.path.join(os.path.dirname(video_path), 'segments'),
                os.path.join(self.output_dir, 'segments'),
                self.output_dir
            ]
            
            for segments_dir in possible_segment_dirs:
                if os.path.exists(segments_dir):
                    try:
                        segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.mp4') and video_name in f]
                        if segment_files:
                            print(f"Found {len(segment_files)} pre-existing segments in {segments_dir}")
                            # Process these segments
                            return self._process_existing_segments(segments_dir, segment_files, video_name, metadata)
                    except Exception as e:
                        print(f"Error checking for segments in {segments_dir}: {str(e)}")
        
            print(f"No pre-existing segments found for {video_path}")
            return []
    
        # Create a unique output directory for this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Created video output directory: {video_output_dir}")
        
        # Create segments directory
        segments_output_dir = os.path.join(os.path.dirname(video_path), '..', 'segments')
        os.makedirs(segments_output_dir, exist_ok=True)
        print(f"Created segments directory: {segments_output_dir}")
        
        # Verify segments directory exists
        if not os.path.exists(segments_output_dir):
            print(f"Error: Failed to create segments directory: {segments_output_dir}")
            # Try an alternative path
            alt_segments_dir = os.path.join(os.path.dirname(video_path), 'segments')
            os.makedirs(alt_segments_dir, exist_ok=True)
            if os.path.exists(alt_segments_dir):
                print(f"Using alternative segments directory: {alt_segments_dir}")
                segments_output_dir = alt_segments_dir
            else:
                print(f"Error: Failed to create alternative segments directory: {alt_segments_dir}")
                return []
    
        print(f"Segments will be stored in: {segments_output_dir}")
        
        # Process the video with sign-segmentation
        if gui:
            update_progress("Processing video", f"Segmenting video {video_path}", 30, 0)
        else:
            print(f"Segmenting video {video_path}")
        
        try:
            # Call the process_video function from segment_video module
            print(f"Calling segment_video.process_video with {video_path}")
            result = self.segment_video.process_video(video_path)
            print(f"Segmentation result: {result}")
            
            # Check if any segments were created
            try:
                segment_files = [f for f in os.listdir(segments_output_dir) if f.endswith('.mp4') and video_name in f]
                print(f"Found {len(segment_files)} segments in {segments_output_dir}")
                
                # Try alternative locations if no segments found
                if not segment_files:
                    alt_dirs = [
                        os.path.join(os.path.dirname(video_path), 'segments'),
                        os.path.join(self.output_dir, 'segments'),
                        video_output_dir
                    ]
                    for alt_dir in alt_dirs:
                        if os.path.exists(alt_dir):
                            try:
                                alt_segment_files = [f for f in os.listdir(alt_dir) if f.endswith('.mp4') and video_name in f]
                                if alt_segment_files:
                                    print(f"Found {len(alt_segment_files)} segments in alternative directory: {alt_dir}")
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
            return self._process_existing_segments(segments_output_dir, segment_files, video_name, metadata)
        except Exception as e:
            print(f"Error segmenting video {video_path}: {str(e)}")
            if gui:
                update_progress(None, f"Error segmenting video: {str(e)}", None, 100)
            return []
    
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
            if any(suffix in base_name for suffix in ['.original.', '.pose.', '.mask.', '.segmentation.']):
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
            
            # Create paths for pose, mask, and segmentation files
            pose_path = os.path.join(segments_dir, f"{segment_name}.pose.mp4")
            mask_path = os.path.join(segments_dir, f"{segment_name}.mask.mp4")
            segmentation_path = os.path.join(segments_dir, f"{segment_name}.segmentation.mp4")
            original_path = os.path.join(segments_dir, f"{segment_name}.original.mp4")
            
            # Create a copy of the original segment file if it doesn't exist
            if not os.path.exists(original_path):
                shutil.copy(segment_path, original_path)
                logger.log_info(f"Created copy of original segment file: {original_path}", segment_name)
            
            # Generate pose visualization if it doesn't exist
            if not os.path.exists(pose_path):
                try:
                    self._generate_pose_visualization(original_path, pose_path)
                    logger.log_info(f"Generated pose visualization: {pose_path}", segment_name)
                except Exception as e:
                    logger.log_error(f"Failed to generate pose visualization", segment_name, e)
                    # Don't continue if pose generation fails
                    continue
            
            # Generate mask if it doesn't exist
            if not os.path.exists(mask_path):
                try:
                    self._generate_mask(original_path, mask_path)
                    logger.log_info(f"Generated mask: {mask_path}", segment_name)
                except Exception as e:
                    logger.log_error(f"Failed to generate mask", segment_name, e)
                    # Don't continue if mask generation fails
                    continue
            
            # Generate segmentation if it doesn't exist
            segmentation_created = False
            if not os.path.exists(segmentation_path):
                # Use refine_mask_v2 if available, otherwise fall back to simple overlay
                if self.refine_mask_v2:
                    try:
                        print(f"Using advanced segmentation with YOLOv11 for segment {segment_name}")
                        logger.log_info(f"Attempting advanced segmentation with YOLOv11", segment_name)
                        
                        # Create a temporary directory for the output
                        temp_output_dir = os.path.join(segments_dir, "temp_seg")
                        os.makedirs(temp_output_dir, exist_ok=True)
                        
                        try:
                            # Process the video using refine_mask_v2
                            success = refine_mask_v2.process_video(original_path, temp_output_dir)
                        except OSError as ose:
                            # Handle the specific OSError: Invalid argument error that occurs with ffmpeg
                            if "[Errno 22] Invalid argument" in str(ose):
                                logger.log_ffmpeg_error(segment_name, "FFmpeg pipe write error when processing mask", ose)
                                success = False
                            else:
                                # Re-raise other OSErrors
                                raise
                        except Exception as e:
                            # Handle any other exceptions that might occur during processing
                            logger.log_error(f"Advanced segmentation failed with error: {str(e)}", segment_name, e)
                            success = False
                        
                        if success:
                            # Find the output file
                            output_files = glob.glob(os.path.join(temp_output_dir, "*.mp4"))
                            if output_files:
                                # Move the first output file to the segmentation path
                                shutil.move(output_files[0], segmentation_path)
                                # Clean up temporary directory
                                shutil.rmtree(temp_output_dir, ignore_errors=True)
                                segmentation_created = True
                                logger.log_segmentation_success(segment_name, "YOLOv11", segmentation_path)
                            else:
                                logger.log_error(f"No output files found in temporary directory", segment_name)
                        else:
                            logger.log_error(f"Advanced segmentation failed", segment_name)
                    except Exception as e:
                        logger.log_segmentation_error(segment_name, "YOLOv11", "Advanced segmentation failed", e)
                        # Clean up temporary directory if it exists
                        if os.path.exists(temp_output_dir):
                            shutil.rmtree(temp_output_dir, ignore_errors=True)
                
                # Fall back to simple overlay if advanced segmentation failed or is not available
                if not segmentation_created:
                    try:
                        logger.log_info(f"Falling back to simple overlay segmentation", segment_name)
                        self._generate_simple_segmentation(original_path, mask_path, segmentation_path)
                        segmentation_created = True
                        logger.log_segmentation_success(segment_name, "simple overlay", segmentation_path)
                    except Exception as e:
                        logger.log_segmentation_error(segment_name, "simple overlay", "Simple segmentation failed", e)
            else:
                segmentation_created = True
                logger.log_info(f"Segmentation file already exists: {segmentation_path}", segment_name)
            
            # Only add to segment_info if all files were created successfully
            if os.path.exists(pose_path) and os.path.exists(mask_path) and segmentation_created:
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
                        "has_pose": os.path.exists(pose_path),
                        "has_mask": os.path.exists(mask_path),
                        "has_segmentation": os.path.exists(segmentation_path) if segmentation_path else False,
                    }
                }
                
                # Add pose, mask, and segmentation paths if they exist
                if os.path.exists(pose_path):
                    segment_info["pose"] = pose_path
                    segment_info["segment_metadata"]["pose"] = os.path.basename(pose_path)
                
                if os.path.exists(mask_path):
                    segment_info["mask"] = mask_path
                    segment_info["segment_metadata"]["mask"] = os.path.basename(mask_path)
                
                if segmentation_path and os.path.exists(segmentation_path):
                    segment_info["segmentation"] = segmentation_path
                    segment_info["segment_metadata"]["segmentation"] = os.path.basename(segmentation_path)
                
                segments_info.append(segment_info)
            else:
                missing_files = []
                if not os.path.exists(pose_path):
                    missing_files.append("pose")
                if not os.path.exists(mask_path):
                    missing_files.append("mask")
                if not segmentation_created:
                    missing_files.append("segmentation")
                
                logger.log_warning(f"Skipping segment due to missing files: {', '.join(missing_files)}", segment_name)
        
        print(f"Successfully processed {len(segments_info)} pre-existing segments")
        return segments_info

    def _generate_pose_visualization(self, input_path, output_path):
        """Generate pose visualization for a video segment.
        
        Args:
            input_path (str): Path to the input video
            output_path (str): Path to save the pose visualization
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.video_to_pose_mask:
            raise ImportError("video_to_pose_mask module not available")
        
        try:
            # Get the directory and base name
            output_dir = os.path.dirname(output_path)
            input_base = os.path.splitext(os.path.basename(input_path))[0]
            
            # Process the video
            self.video_to_pose_mask.process_video(input_path, output_dir)
            
            # Find the pose file
            pose_file = os.path.join(output_dir, f"{input_base}_pose.mp4")
            
            # Move to the desired output path
            if os.path.exists(pose_file):
                shutil.move(pose_file, output_path)
                return True
            else:
                raise FileNotFoundError(f"Pose file not found: {pose_file}")
        except Exception as e:
            logger.log_error(f"Failed to generate pose visualization: {str(e)}", os.path.basename(input_path), e)
            return False
    
    def _generate_mask(self, input_path, output_path):
        """Generate mask for a video segment.
        
        Args:
            input_path (str): Path to the input video
            output_path (str): Path to save the mask
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.video_to_pose_mask:
            raise ImportError("video_to_pose_mask module not available")
        
        try:
            # Get the directory and base name
            output_dir = os.path.dirname(output_path)
            input_base = os.path.splitext(os.path.basename(input_path))[0]
            
            # Process the video (if not already processed for pose)
            if not os.path.exists(os.path.join(output_dir, f"{input_base}_mask.mp4")):
                self.video_to_pose_mask.process_video(input_path, output_dir)
            
            # Find the mask file
            mask_file = os.path.join(output_dir, f"{input_base}_mask.mp4")
            
            # Move to the desired output path
            if os.path.exists(mask_file):
                shutil.move(mask_file, output_path)
                return True
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_file}")
        except Exception as e:
            logger.log_error(f"Failed to generate mask: {str(e)}", os.path.basename(input_path), e)
            return False
    
    def _generate_simple_segmentation(self, original_path, mask_path, output_path):
        """Generate a simple segmentation by overlaying the mask on the original video.
        
        Args:
            original_path (str): Path to the original video
            mask_path (str): Path to the mask video
            output_path (str): Path to save the segmentation
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if input files exist
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"Original video not found: {original_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask video not found: {mask_path}")
            
            # Create the overlay using ffmpeg
            overlay_cmd = [
                'ffmpeg', '-y',
                '-i', original_path,
                '-i', mask_path,
                '-filter_complex', '[0:v][1:v]overlay=format=auto:alpha=0.5',
                '-c:v', 'libx264',
                output_path
            ]
            
            # Run the command
            process = subprocess.run(overlay_cmd, check=True, capture_output=True, text=True)
            
            # Check if the output file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            else:
                raise RuntimeError(f"Failed to create segmentation: {process.stderr}")
        except Exception as e:
            logger.log_error(f"Failed to generate simple segmentation: {str(e)}", os.path.basename(original_path), e)
            return False


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
                    if "segment_path" not in segment_info and "original" in segment_info:
                        segment_path = segment_info["original"]
                    elif "segment_path" in segment_info:
                        segment_path = segment_info["segment_path"]
                    else:
                        print(f"Warning: No path found for segment {segment_name}. Skipping.")
                        continue
                    
                    # Get the base segment name without any extensions
                    base_segment_name = os.path.splitext(os.path.basename(segment_path))[0]
                    
                    # Remove any existing subcategory suffixes
                    for suffix in ['.original', '.pose', '.mask', '.segmentation']:
                        if base_segment_name.endswith(suffix):
                            base_segment_name = base_segment_name[:-len(suffix)]
                    
                    # Define the paths for all file types
                    original_path = os.path.join(os.path.dirname(segment_path), f"{base_segment_name}.original.mp4")
                    pose_path = os.path.join(os.path.dirname(segment_path), f"{base_segment_name}.pose.mp4")
                    mask_path = os.path.join(os.path.dirname(segment_path), f"{base_segment_name}.mask.mp4")
                    segmentation_path = os.path.join(os.path.dirname(segment_path), f"{base_segment_name}.segmentation.mp4")
                    
                    # Create a sample with all available files
                    sample = {}
                    
                    # Add the original video
                    if os.path.exists(original_path) and os.path.getsize(original_path) > 0:
                        sample["mp4"] = original_path
                    elif os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                        sample["mp4"] = segment_path
                    else:
                        print(f"Warning: Original video file not found for {segment_name}. Skipping.")
                        continue
                    
                    # Add the pose video if available
                    if os.path.exists(pose_path) and os.path.getsize(pose_path) > 0:
                        sample["pose.mp4"] = pose_path
                        segment_info["segment_metadata"]["pose"] = f"{base_segment_name}.pose.mp4"
                    else:
                        segment_info["segment_metadata"]["pose"] = None
                    
                    # Add the mask video if available
                    if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                        sample["mask.mp4"] = mask_path
                        segment_info["segment_metadata"]["mask"] = f"{base_segment_name}.mask.mp4"
                    else:
                        segment_info["segment_metadata"]["mask"] = None
                    
                    # Add the segmentation video if available
                    if os.path.exists(segmentation_path) and os.path.getsize(segmentation_path) > 0:
                        sample["segmentation.mp4"] = segmentation_path
                        segment_info["segment_metadata"]["segmentation"] = f"{base_segment_name}.segmentation.mp4"
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
                                    print(f"Warning: File {path} has zero size. Skipping.")
                                    continue
                                
                                # Create tar info
                                info = tarfile.TarInfo(f"{segment_name}_{i}.{key}")
                                info.size = file_size
                                
                                # Add file with binary mode to avoid encoding issues
                                with open(path, 'rb') as file_data:
                                    tar.addfile(info, file_data)
                            except Exception as e:
                                print(f"Error adding file {path} to tar: {e}")
            
            shard_paths.append(shard_path)
        
        return shard_paths


def update_segments_info(segments_info):
    """Update segments_info with the actual paths to the processed files."""
    for segment in segments_info:
        segment_name = segment["segment_name"]
        original_path = segment["original"]
        
        # Get the directory containing the segment files
        segments_dir = os.path.dirname(original_path)
        
        # Check for pose, mask, and segmentation files
        pose_path = os.path.join(segments_dir, f"{segment_name}.pose.mp4")
        mask_path = os.path.join(segments_dir, f"{segment_name}.mask.mp4")
        segmentation_path = os.path.join(segments_dir, f"{segment_name}.segmentation.mp4")
        
        # Update segment info with actual paths
        if os.path.exists(pose_path):
            segment["pose"] = pose_path
            segment["segment_metadata"]["has_pose"] = True
        
        if os.path.exists(mask_path):
            segment["mask"] = mask_path
            segment["segment_metadata"]["has_mask"] = True
        
        if os.path.exists(segmentation_path):
            segment["segmentation"] = segmentation_path
            segment["segment_metadata"]["has_segmentation"] = True
    
    return segments_info


def update_progress(operation, message, overall_progress=None, op_progress=None):
    """Update the GUI progress if it's running."""
    global gui
    if gui:
        if message:
            update_gui(gui, {"type": "log", "text": message})
        if operation:
            update_gui(gui, {"type": "operation", "text": operation})
        if overall_progress is not None:
            update_gui(gui, {"type": "overall_progress", "value": overall_progress})
        if op_progress is not None:
            update_gui(gui, {"type": "op_progress", "value": op_progress})
        
        # Check if the process has been cancelled
        if hasattr(gui, 'cancelled') and gui.cancelled:
            update_gui(gui, {"type": "log", "text": "Processing cancelled by user."})
            return True
    return False

def process_with_gui(args):
    """Process videos with GUI progress updates."""
    try:
        # Initialize directories
        os.makedirs(args.output_dir, exist_ok=True)
        downloads_dir = os.path.join(args.output_dir, "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        processed_dir = os.path.join(args.output_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        update_progress("Initializing", "Starting video processing pipeline...", 0, 0)
        
        # Step 1: Download videos
        update_progress("Downloading videos", "=== Step 1: Downloading videos from DBL-sign ===", 10, 0)
        downloader = DBLSignDownloader(downloads_dir)
        video_info_list = downloader.download_videos(args.num_videos, args.language_code, args.project_name)
        
        # Check for cancellation
        if update_progress(None, f"Downloaded {len(video_info_list)} videos", 20, 100):
            return
        
        # Step 2: Process videos with sign-segmentation
        update_progress("Processing videos", "=== Step 2: Processing videos with sign-segmentation ===", 30, 0)
        processor = SignSegmentationProcessor(processed_dir)
        
        all_segments = []
        for i, video_info in enumerate(video_info_list):
            # Extract video path and metadata from the dictionary
            video_path = video_info["path"]
            metadata = video_info["metadata"]
            
            # Check for cancellation
            if update_progress(None, f"Processing video {i+1}/{len(video_info_list)}: {os.path.basename(video_path)}", 
                           30 + (i / len(video_info_list)) * 40, 0):
                return
            
            video_progress = (i / len(video_info_list)) * 100
            update_progress(None, f"Processing video {i+1}/{len(video_info_list)}: {os.path.basename(video_path)}", 
                           30 + (i / len(video_info_list)) * 40, video_progress)
            
            segments = processor.process_video(video_path, metadata)
            all_segments.extend(segments)
            
            # Check for cancellation after each video
            if update_progress(None, f"Completed processing video {i+1}/{len(video_info_list)}", 
                           30 + ((i+1) / len(video_info_list)) * 40, 100):
                return
        
        # Step 3: Create WebDataset
        update_progress("Creating WebDataset", "=== Step 3: Creating WebDataset ===", 70, 0)
        webdataset_dir = os.path.join(args.output_dir, "webdataset")
        os.makedirs(webdataset_dir, exist_ok=True)
        
        # Check for cancellation
        if update_progress(None, "Creating WebDataset...", 70, 50):
            return
        
        # Create WebDataset
        creator = WebDatasetCreator(webdataset_dir)
        creator.create_webdataset(all_segments, shard_size=args.shard_size)
        
        # Check for cancellation
        if update_progress(None, "WebDataset created", 80, 100):
            return
        
        # Step 4: Create manifest
        update_progress("Creating manifest", "=== Step 4: Creating manifest ===", 90, 0)
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        
        # Update segments_info with actual paths to processed files
        all_segments = update_segments_info(all_segments)
        
        # Check for cancellation
        if update_progress(None, "Creating manifest file...", 90, 50):
            return
        
        # Create manifest
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({"segments": all_segments}, f, indent=2)
        
        update_progress("Complete", f"Processing complete! Manifest saved to {manifest_path}", 100, 100)
    except Exception as e:
        update_progress("Error", f"Error during processing: {str(e)}", 100, 100)
        raise e

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
    video_info_list = downloader.download_videos(args.num_videos, args.language_code, args.project_name)
    
    # Step 2: Process videos with sign-segmentation
    print("=== Step 2: Processing videos with sign-segmentation ===")
    processor = SignSegmentationProcessor(processed_dir)
    
    all_segments = []
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
    
    # Update segments_info with actual paths to processed files
    all_segments = update_segments_info(all_segments)
    
    # Create manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({"segments": all_segments}, f, indent=2)
    
    print(f"Processing complete! Manifest saved to {manifest_path}")

def main():
    """Main function to run the entire process."""
    parser = argparse.ArgumentParser(description="Prepare sign language videos for HuggingFace datasets")
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to download")
    parser.add_argument("--language-code", type=str, help="Filter by language code")
    parser.add_argument("--project-name", type=str, help="Filter by project name")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of segments per WebDataset shard")
    parser.add_argument("--with-gui", action="store_true", help="Show progress GUI")
    args = parser.parse_args()
    
    # Start GUI if requested
    global gui, root
    if args.with_gui:
        root, gui = start_gui()
        # Start processing in a separate thread
        threading.Thread(target=process_with_gui, args=(args,), daemon=True).start()
        root.mainloop()
    else:
        # Run processing directly
        process_without_gui(args)

if __name__ == "__main__":
    main()
