#!/bin/bash
set -euo pipefail  # safer bash settings: stop on error, undefined vars, or failed pipes

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go two levels up to find the project root
project_root="$(dirname "$(dirname "$script_dir")")"

# Parse arguments
dir_to_search="${1:-$project_root/downloads}"  # default if not given
workers="${2:-16}"                             # default to 16 workers
csv_path=${3:-"$script_dir/folder_durations.csv"}


# Debug printout (remove or comment out if undesired)
echo "Project root: $project_root"
echo "Directory to search: $dir_to_search"
echo "Parallel workers: $workers"

# Find all .mp4 files not containing "animation", run ffprobe in parallel, and sum durations
total_duration=$(find "$dir_to_search" -type f -name "*.mp4" |
  grep -v "animation" |
  parallel --no-notice --progress -j "$workers" \
    ffprobe -v error -select_streams v:0 \
            -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 {} |
  awk '{s+=$1} END {print s}')

# Format the duration in hh:mm:ss
hours=$(awk -v t="$total_duration" 'BEGIN { printf "%02d", int(t / 3600) }')
minutes=$(awk -v t="$total_duration" 'BEGIN { printf "%02d", int((t % 3600) / 60) }')
seconds=$(awk -v t="$total_duration" 'BEGIN { printf "%05.2f", t % 60 }')

formatted="${hours}:${minutes}:${seconds}"

echo "Total duration: $total_duration seconds"
echo "Formatted: $formatted (hh:mm:ss)"

# CSV output
# csv_path="$script_dir/folder_durations.csv"

# If CSV doesn't exist, write the header
if [ ! -f "$csv_path" ]; then
  echo "dir,total_duration_sec,duration_formatted" > "$csv_path"
fi

# Append this run's results
echo "$dir_to_search,$total_duration,$formatted" >> "$csv_path"

echo "Appended to: $csv_path"

