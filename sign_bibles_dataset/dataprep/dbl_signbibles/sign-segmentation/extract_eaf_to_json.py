import argparse
import json
import logging
from pathlib import Path

import pympi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eaf-segmenter")


def extract_sentence_sign_segments(eaf_path: Path) -> Path:
    """
    Extract SENTENCE and SIGN annotations from an EAF file
    and save to a .autosegmented_segments.json file, grouped by tier.
    """
    output_json = eaf_path.with_name(eaf_path.stem + ".autosegmented_segments.json")

    eaf = pympi.Elan.Eaf(str(eaf_path))
    tier_names = eaf.get_tier_names()

    data = {}

    for tier in ["SENTENCE", "SIGN"]:
        if tier not in tier_names:
            logger.warning(f"{eaf_path.name}: {tier} tier not found.")
            data[tier] = []
            continue

        annotations = eaf.get_annotation_data_for_tier(tier)
        data[tier] = [
            {"start_ms": int(start), "end_ms": int(end), "text": text.strip() if text else ""}
            for start, end, text in annotations
        ]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total = sum(len(anns) for anns in data.values())
    logger.debug(f"{eaf_path.name}: Saved {total} segments to {output_json.name}")
    return output_json


def recursive_eaf_to_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        eaf_files = [path]
    elif path.is_dir():
        eaf_files = list(path.rglob("*.model*.eaf"))
    else:
        raise ValueError(f"Path must be a file or directory: {path}")

    logger.info(f"Found {len(eaf_files)} EAF file(s) in {path}")

    for eaf_file in tqdm(eaf_files, desc="Processing .eaf files"):
        extract_sentence_sign_segments(eaf_file)


def main():
    parser = argparse.ArgumentParser(description="Extract SENTENCE and SIGN segments from EAF files.")
    parser.add_argument("path", type=Path, help="Path to a .eaf file or a directory containing .eaf files")
    args = parser.parse_args()

    path = args.path.resolve()
    recursive_eaf_to_json(path)


if __name__ == "__main__":
    main()
