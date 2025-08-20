#!/bin/bash
set -euo pipefail

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$(dirname "$script_dir")")"

# Parse arguments
dir_to_search="${1:-$project_root/downloads}"  # e.g. sign-bibles-dataset-script-downloads/
depth="${2:-1}"                                # 1: ase/esl/...  2: projects inside
workers="${3:-8}"                              # parallel jobs
rows="${4:-6}"      # number of rows in grid
cols="${5:-6}"      # number of columns in grid
faces_per_dir=$((rows * cols))  # override faces_per_dir from grid size


# Debug printout (remove or comment out if undesired)
echo "Project root: $project_root"
echo "Script Dir: $script_dir"
echo "Directory to search: $dir_to_search"
echo "Depth: $depth"
echo "Parallel workers: $workers"
echo "Rows: $rows"
echo "Cols: $cols"
echo "Faces Per Dir: $faces_per_dir"



# Output directory
out_dir="$script_dir/random_facecrops/depth_$depth"
mkdir -p "$out_dir"

# --------- Function 1: Find and list facecrop paths ----------
facecrop_finder() {
  dir="$1"
  out_dir="$2"
  out_subdir="$out_dir/$(basename "$dir")"
  out_file="$out_subdir/facecrop_paths.txt"

  # Skip if already processed
  [[ -f "$out_file" ]] && return

  mkdir -p "$out_subdir"
  find "$dir" -wholename "*/facecrops/*.png" > "$out_file"
}
export -f facecrop_finder

# --------- Function 2: Shuffle and copy N faces ----------
facecrop_selector() {
  dir="$1"
  out_dir="$2"
  count="$3"
  subdir="$(basename "$dir")"
  input_file="$out_dir/$subdir/facecrop_paths.txt"
  dest_dir="$out_dir/$subdir/samples"

  [[ ! -f "$input_file" ]] && echo "Missing $input_file" && return

  mkdir -p "$dest_dir"

  shuf "$input_file" | head -n "$count" | while read -r src; do
    # Get parent and grandparent directories
    parent="$(basename "$(dirname "$src")")"
    grandparent="$(basename "$(dirname "$(dirname "$src")")")"
    filename="$(basename "$src")"
    
    # Construct new filename
    new_filename="${grandparent}_${parent}_${filename}"
    
    cp "$src" "$dest_dir/$new_filename"
  done
}

export -f facecrop_selector
make_image_grid() {
  dir="$1"
  out_dir="$2"
  rows="$3"
  cols="$4"
  subdir="$(basename "$dir")"
  sample_dir="$out_dir/$subdir/samples"
  grid_path="$out_dir/$subdir/grid_$rows_$cols.png"

  [[ -d "$sample_dir" ]] || return
  [[ -f "$grid_path" ]] && return

  montage "$sample_dir"/*.png \
    -resize 48x48 \
    -tile "${cols}x" \
    -geometry 48x48+2+2 \
    "$grid_path"
}

export -f make_image_grid




# --------- Step 1: Ensure all output dirs exist ----------
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d | \
  parallel -j "$workers" mkdir -p "$out_dir/{/}"

# --------- Step 2: Find all facecrops ----------
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d | \
  parallel -j "$workers" facecrop_finder {} "$out_dir"

# --------- Step 3: Select and copy N random crops ----------
find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d | \
  parallel -j "$workers" facecrop_selector {} "$out_dir" "$faces_per_dir"

# --------- Step 4: Make image grid from copied samples ----------
# find "$dir_to_search" -mindepth "$depth" -maxdepth "$depth" -type d | \
#   parallel -j "$workers" make_image_grid {} "$out_dir" "$rows" "$cols"


