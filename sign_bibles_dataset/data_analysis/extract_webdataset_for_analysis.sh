#!/bin/bash
set -euo pipefail

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$(dirname "$script_dir")")"

# Parse arguments
src_root="${1:-/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset}"
dst_root="${2:-/data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted}"
workers="${3:-64}"  # Parallel jobs

# Debug printout
echo "Script directory: $script_dir"
echo "Project root: $project_root"
echo "Source root: $src_root"
echo "Destination root: $dst_root"
echo "Parallel workers: $workers"

# Function to extract a tar file
extract_tar() {
    tar_path="$1"
    src_root="$2"
    dst_root="$3"

    # Compute relative path
    rel_path="${tar_path#"$src_root"/}"
    rel_dir="$(dirname "$rel_path")"
    base_name="$(basename "$tar_path" .tar)"
    
    # Destination directory: same structure, without the .tar
    out_dir="$dst_root/$rel_dir"
    mkdir -p "$out_dir"

    echo "Extracting: $tar_path --> $out_dir"

    # Extract directly into the destination dir
    tar -xvf "$tar_path" -C "$out_dir"
}

export -f extract_tar

# Use find and GNU parallel
find "$src_root" -type f -wholename '*ase/*.tar' | sort | \
  parallel --progress -j "$workers" \
    extract_tar {} "$src_root" "$dst_root"
