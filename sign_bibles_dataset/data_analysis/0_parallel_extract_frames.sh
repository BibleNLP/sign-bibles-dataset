#!/bin/bash
set -euo pipefail  # safer bash settings: stop on error, undefined vars, or failed pipes

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go two levels up to find the project root
project_root="$(dirname "$(dirname "$script_dir")")"

# Parse arguments
dir_to_search="${1:-$project_root/downloads}"  # default if not given
workers="${2:-16}"                         # default to 16 workers
step="${3:-10s}"                          # default step is 10s

# Debug printout (remove or comment out if undesired)
echo "Project root: $project_root"
echo "Directory to search: $dir_to_search"
echo "Parallel workers: $workers"
echo "Step size: $step"

# Run the command in parallel
find "$dir_to_search" -name "*.mp4" |
  grep -v "animation" |
  parallel --progress -j "$workers" \
    python "$project_root/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/ocr/extract_frame.py" "{}" --step "$step" --quiet
