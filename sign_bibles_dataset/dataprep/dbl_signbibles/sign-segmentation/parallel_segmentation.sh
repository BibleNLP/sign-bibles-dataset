#!/bin/bash
set -euo pipefail
search_dir="$1"
workers="$2"
find "$search_dir" -type f -name "*.pose"|parallel --progress -j"$workers" python sign_bibles_dataset/dataprep/dbl_signbibles/sign-segmentation/recursively_run_segmentation.py 