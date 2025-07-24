#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def find_npz_missing_keys(root: Path, required_keys: list[str]) -> list[Path]:
    """Return a list of .npz file paths that are missing any of the required keys."""
    missing_files = []

    print(f"Looking for keys {required_keys}")

    for file_path in tqdm(root.rglob("*.npz"), desc="Checking files for keys"):
        with np.load(file_path, allow_pickle=False) as data:
            if not all(key in data for key in required_keys):
                missing_files.append(file_path)

    return missing_files


def find_npz_missing_keys_or_empty(
    root: Path, required_keys: list[str], expected_keypoints_count: int = 134
) -> list[Path]:
    """
    Return a list of .npz file paths that are missing any of the required keys
    or contain empty arrays for those keys.
    """
    missing_or_empty = []

    for file_path in tqdm(root.rglob("*.npz"), desc="Checking files"):
        try:
            with np.load(file_path, allow_pickle=False) as data:
                for key in required_keys:
                    if key not in data or data[key].size == 0 or data[key].shape[2] != expected_keypoints_count:
                        missing_or_empty.append(file_path)
                        break  # Skip further checking this file
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            missing_or_empty.append(file_path)

    return missing_or_empty


def main():
    parser = argparse.ArgumentParser(description="Find .npz files missing specified keys.")
    parser.add_argument("path", type=Path, help="Root directory to search recursively.")
    parser.add_argument("--keys", nargs="+", required=True, help="List of required keys.")
    parser.add_argument("--print-list", action="store_true", help="Print the whole list")
    args = parser.parse_args()

    missing = find_npz_missing_keys_or_empty(args.path, args.keys)
    if missing:
        print(f"Total Files Missing 1 or more {args.keys}: {len(missing)}")
        if args.print_list:
            print("\nMissing keys in the following files:")
            for path in missing:
                print(path)
    else:
        print("All .npz files contain the specified keys and nonempty arrays")


if __name__ == "__main__":
    main()
