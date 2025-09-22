#!/usr/bin/env python3
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def process_file(path: Path) -> tuple[bool, bool, int, list[int]]:
    """
    Process a JSON file and return:
    - has_nonempty_text: True if any element has non-empty 'text'
    - list_longer_than_one: True if top-level list has > 1 element
    - length: number of elements in the top-level list (0 if not a list)
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return False, False, 0

        length = len(data)
        vrefs = []
        for item in data:
            vrefs.extend(item.get("biblenlp-vref", []))
        has_nonempty_text = any(isinstance(item, dict) and item.get("text", "") != "" for item in data)
        list_longer_than_one = length > 1

        return has_nonempty_text, list_longer_than_one, length, vrefs

    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Failed to read %s: %s", path, e)
        return False, False, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan JSON files for non-empty text and list length > 1")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to search for JSON files",
    )
    args = parser.parse_args()

    json_files = list(args.root.rglob("*transcripts*.json"))
    logging.info("Found %d JSON files", len(json_files))

    # global counts
    nonempty_text_count = 0
    longer_than_one_count = 0
    all_vrefs = []

    # stats by (language, project)
    stats: dict[str, dict[str, int, list]] = defaultdict(
        lambda: {"files": 0, "nonempty_text": 0, "longer_than_one": 0, "total_verse_count": 0, "vrefs": []}
    )

    for path in tqdm(json_files, desc="Processing JSON files"):
        has_text, longer_than_one, length, path_vrefs = process_file(path)

        # update *global* counters once per file
        if has_text:
            nonempty_text_count += 1
        if longer_than_one:
            longer_than_one_count += 1

        # folders
        project_folder = path.parent.name
        language_code_folder = path.parent.parent.name
        all_vrefs.extend(path_vrefs)

        # update *both* language and project buckets
        for key in [language_code_folder, f"{language_code_folder}/{project_folder}"]:
            stats[key]["files"] += 1
            stats[key]["total_verse_count"] += length
            if has_text:
                stats[key]["nonempty_text"] += 1
            if longer_than_one:
                stats[key]["longer_than_one"] += 1
            stats[key]["vrefs"].extend(path_vrefs)

    # print global stats
    print(f"Total files: {len(json_files)}")
    print(f"Files with non-empty text: {nonempty_text_count}")
    print(f"Files with list length > 1: {longer_than_one_count}\n")
    print(f"All vrefs: {len(all_vrefs)}. Set: {len(set(all_vrefs))}")

    # print stats by language/project
    print("Stats by language and project:")
    for key, values in sorted(stats.items()):
        total_verse_count = values["total_verse_count"]
        mean_of_all_files = total_verse_count / values["files"]
        mean_of_nonempty = total_verse_count / values["nonempty_text"] if values["nonempty_text"] > 0 else 0
        vrefs = values["vrefs"]
        print(
            f"{key}: "
            f"\n\t{values['files']} files, "
            f"{values['nonempty_text']} with non-empty text, "
            f"{values['longer_than_one']} with list > 1. "
            f"\n\tTotal segment count across all videos: {total_verse_count}. "
            f"\n\tAverage segment count of all: {mean_of_all_files:.2f}, of nonempty: {mean_of_nonempty:.2f}"
            f"\n\tVrefs count: {len(vrefs)}, set count: {len(set(vrefs))}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


#  python count_transcripts_in_extracted_webdataset.py /data/petabyte/cleong/data/DBL_Deaf_Bibles/webdataset_extracted/ > count_transcripts_and_segments/transcript_counts_and_stats.txt
