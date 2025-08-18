#!/bin/bash
set -euo pipefail  # safer bash settings: stop on error, undefined vars, or failed pipes

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go two levels up to find the project root
project_root="$(dirname "$(dirname "$script_dir")")"

# Parse arguments
dir_to_search="${1:-$project_root/downloads}"  # default if not given
workers="${2:-16}"                         # default to 16 workers

# Debug printout (remove or comment out if undesired)
echo "Project root: $project_root"
echo "Directory to search: $dir_to_search"
echo "Parallel workers: $workers"

# Run the command in parallel. The python script automatically looks for {}_frames/
find "$dir_to_search" -type d -name "*_frames" |
  grep -v "animation" |
  parallel --progress -j "$workers" \
    python "$project_root/sign_bibles_dataset/data_analysis/extract_faces_from_frames_using_poses.py" "{.}"
