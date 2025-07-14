#!/bin/bash
set -euo errexit
poses_dir="$1"
workers="$2"
segment_type="SENTENCE"
segments_dir="$poses_dir/segments"
mkdir -p segments_dir
# find "$poses_dir" -type f -name "*.pose"|grep -v "SENTENCE"|grep -v "SIGN"| parallel --progress -j "$workers" pose_to_segments --pose="{}" --save-segments "$segment_type" --elan="{.}.eaf" --video="{.}.mp4"
find "$poses_dir" -type f -name "*.pose"|grep -v "SENTENCE"|grep -v "SIGN"| parallel --progress -j "$workers" pose_to_segments --pose="{}" --elan="{.}.eaf" --video="{.}.mp4"
find "$poses_dir" -name "*$segment_type*.pose" |parallel --progress -x mv "{}" "$segments_dir"
