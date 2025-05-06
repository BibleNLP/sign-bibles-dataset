import os
import subprocess
import cv2
import mediapipe as mp
import numpy as np
from vidgear.gears import WriteGear
import pympi
import shutil
import json
import argparse
import boto3
from pathlib import Path
import tempfile

def download_from_s3(bucket_name, key, local_path):
    """Download a file from S3 to a local path."""
    s3_client = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3_client.download_file(bucket_name, key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def process_video(input_video_path):
    # Create segments directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(input_video_path), '..', 'segments')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # First, run video_to_pose to get pose data
    pose_file = os.path.join(output_dir, f"{base_filename}.pose")
    subprocess.run(['video_to_pose', '-i', input_video_path, '-o', pose_file])
    print("Pose extraction completed")
    
    # Then, run pose_to_segments to get ELAN file
    eaf_file = os.path.join(output_dir, f"{base_filename}.eaf")
    # Remove existing ELAN file if it exists
    if os.path.exists(eaf_file):
        os.remove(eaf_file)
    subprocess.run(['pose_to_segments', '-i', pose_file, '-o', eaf_file, '--video', input_video_path])
    print("Segmentation completed")
    
    # Read ELAN file and extract segments
    eaf = pympi.Elan.Eaf(eaf_file)
    print("Available tiers:", list(eaf.get_tier_names()))
    
    # Get annotations from SENTENCE tier
    annotations = []
    if 'SENTENCE' in eaf.get_tier_names():
        annotations = eaf.get_annotation_data_for_tier('SENTENCE')
    
    print(f"Found {len(annotations)} annotations")
    
    # Sort annotations by start time
    annotations.sort(key=lambda x: x[0])
    
    # Process gaps and create segments
    segments = []
    current_time = 0
    
    # Get video duration
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = int((total_frames / fps) * 1000)  # Duration in milliseconds
    cap.release()
    
    # Process all annotations and gaps
    for i, (start_time, end_time, text) in enumerate(annotations):
        # If there's a gap between current_time and start_time, create 15-second segments
        while current_time < start_time:
            segment_end = min(current_time + 15000, start_time)  # 15000ms = 15s
            segments.append((current_time, segment_end, ""))
            current_time = segment_end
        
        # Add the current annotation segment
        segments.append((start_time, end_time, text))
        current_time = end_time
    
    # Handle any remaining time after the last annotation
    while current_time < video_duration:
        segment_end = min(current_time + 15000, video_duration)
        segments.append((current_time, segment_end, ""))
        current_time = segment_end
    
    print(f"Created {len(segments)} segments")
    
    # Merge short segments with adjacent ones
    MIN_DURATION = 5000  # milliseconds
    merged_segments = []
    i = 0
    
    while i < len(segments):
        current_start, current_end, current_text = segments[i]
        current_duration = current_end - current_start
        
        # If segment is long enough, add it as is
        if current_duration >= MIN_DURATION:
            merged_segments.append((current_start, current_end, current_text))
            i += 1
            continue
        
        # If this is the last segment and it's too short, merge it with the previous one
        if i == len(segments) - 1 and merged_segments:
            prev_start, prev_end, prev_text = merged_segments.pop()
            merged_text = prev_text if not current_text else f"{prev_text} {current_text}".strip()
            merged_segments.append((prev_start, current_end, merged_text))
            break
        
        # Otherwise, merge with the next segment
        if i < len(segments) - 1:
            next_start, next_end, next_text = segments[i + 1]
            merged_text = current_text if not next_text else f"{current_text} {next_text}".strip()
            merged_segments.append((current_start, next_end, merged_text))
            i += 2
        
        i += 1
    
    print(f"Merged into {len(merged_segments)} segments")
    
    # Extract video segments
    output_pattern = os.path.join(output_dir, f"{base_filename}_segment_%04d.mp4")
    
    # Get video info
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json',
        input_video_path
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    video_info = json.loads(probe_result.stdout)
    
    # Calculate minimum segment duration (5 seconds)
    MIN_DURATION = 5000  # milliseconds
    
    # Process segments
    valid_segments = []
    for i, (start_ms, end_ms, text) in enumerate(merged_segments):
        # Skip segments that are too short
        if end_ms - start_ms < MIN_DURATION:
            continue
            
        output_file = output_pattern % (len(valid_segments) + 1)
        
        # Convert milliseconds to seconds for ffmpeg
        start_s = start_ms / 1000
        duration_s = (end_ms - start_ms) / 1000
        
        # Use ffmpeg to extract segment, forcing keyframes and re-encoding
        command = [
            'ffmpeg', '-y',
            '-i', input_video_path,
            '-ss', str(start_s),
            '-t', str(duration_s),
            '-force_key_frames', f"expr:gte(t,{start_s})",
            '-c:v', 'libx264',  # Re-encode video
            '-preset', 'fast',   # Fast encoding
            '-crf', '23',       # Good quality
            '-c:a', 'aac',      # Re-encode audio
            output_file
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Verify the output file exists and has content
        if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:  # Minimum 10KB
            valid_segments.append((start_ms, end_ms, text))
            print(f"Created segment {len(valid_segments)}/{len(merged_segments)}: {start_ms}ms to {end_ms}ms")
        else:
            print(f"Warning: Failed to create valid segment for {start_ms}ms to {end_ms}ms")
            if os.path.exists(output_file):
                os.remove(output_file)
    
    print(f"Successfully created {len(valid_segments)} valid segments")

def process_input(input_source):
    """Process input source which can be a file, directory, or S3 bucket."""
    # Check if input is an S3 path (s3://bucket-name/key)
    if input_source.startswith('s3://'):
        # Parse S3 URL
        parts = input_source[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError("Invalid S3 URL format. Use s3://bucket-name/key")
        
        bucket_name, key = parts
        # Create temporary directory for downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            if key.endswith(('/','')): # If it's a prefix (directory)
                s3_client = boto3.client('s3')
                paginator = s3_client.get_paginator('list_objects_v2')
                
                # List all objects with the given prefix
                for page in paginator.paginate(Bucket=bucket_name, Prefix=key):
                    for obj in page.get('Contents', []):
                        if obj['Key'].endswith(('.mp4', '.avi', '.mov')):
                            local_path = os.path.join(temp_dir, os.path.basename(obj['Key']))
                            if download_from_s3(bucket_name, obj['Key'], local_path):
                                print(f"\nProcessing {obj['Key']}...")
                                process_video(local_path)
            else: # Single file
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
            if input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                print(f"\nProcessing {input_path.name}...")
                process_video(str(input_path))
            else:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")
        
        elif input_path.is_dir():
            for file_path in input_path.glob('*'):
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    print(f"\nProcessing {file_path.name}...")
                    process_video(str(file_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video files for sign language segmentation')
    parser.add_argument('input', help='Input source: can be a file path, directory path, or S3 URL (s3://bucket-name/key)')
    args = parser.parse_args()
    
    process_input(args.input)
