#!/bin/bash
set -euo pipefail  # safer bash settings: stop on error, undefined vars, or failed pipes

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
dir_to_search="${1:-$project_root/downloads}"   # default if not given
skintone_to_look_for="$2"

ag -l "$skintone_to_look_for" "$dir_to_search"|grep result.csv