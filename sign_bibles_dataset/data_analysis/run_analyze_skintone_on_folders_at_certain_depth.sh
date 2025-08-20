#!/bin/bash
set -euo pipefail  # safer bash settings: stop on error, undefined vars, or failed pipes

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"





# Go two levels up to find the project root
project_root="$(dirname "$(dirname "$script_dir")")"

# Parse arguments
dir_to_search="${1:-$project_root/downloads}"  # default if not given
depth="${2:-1}"                     # how deep to search, default is 1 so ase/esl and so on. 2 is the project lavel
workers="${3:-8}"                         # default to 8 workers

# Debug printout (remove or comment out if undesired)
echo "Project root: $project_root"
echo "Directory to search: $dir_to_search"
echo "Depth to search: $depth"
echo "Parallel workers: $workers"

#output dir
out_dir="$script_dir/skintone_plots/depth_$depth"

mkdir -p "$out_dir"
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d|parallel -j "$workers" python analyze_skintone.py "{}" --output-dir "$out_dir/{/}" --tight
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d|parallel -j "$workers" python analyze_skintone.py "{}" --output-dir "$out_dir/{/}"
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d|parallel -j "$workers" python analyze_skintone.py "{}" --output-dir "$out_dir/{/}" --by-file
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d|parallel -j "$workers" python analyze_skintone.py "{}" --output-dir "$out_dir/{/}" --by-file --tight