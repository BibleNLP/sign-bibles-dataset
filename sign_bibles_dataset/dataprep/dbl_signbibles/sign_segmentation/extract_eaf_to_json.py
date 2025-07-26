import argparse
import json
import logging
from pathlib import Path

import pympi
from pose_format import Pose
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eaf-segmenter")


def extract_sentence_sign_segments(eaf_path: Path) -> Path:
    """
    Extract SENTENCE and SIGN annotations from an EAF file
    and save to a .autosegmented_segments.json file, grouped by tier.
    """
    output_json = eaf_path.with_name(eaf_path.stem + ".autosegmented_segments.json")
    input_pose_path = eaf_path.parent / (eaf_path.name.split(".")[0] + ".pose")

    fps = None
    if input_pose_path.is_file():
        pose = Pose.read(input_pose_path.read_bytes())
        fps = pose.body.fps

    eaf = pympi.Elan.Eaf(str(eaf_path))
    tier_names = eaf.get_tier_names()

    data = {}

    for tier in ["SENTENCE", "SIGN"]:
        if tier not in tier_names:
            logger.warning(f"{eaf_path.name}: {tier} tier not found.")
            data[tier] = []
            continue

        annotations = eaf.get_annotation_data_for_tier(tier)
        tier_data = []
        for start_ms, end_ms, text in annotations:
            segment = {
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "text": text.strip() if text else "",
            }

            if fps is not None:
                # Add frame index equivalents
                segment["start_frame"] = round(start_ms / 1000 * fps)
                segment["end_frame"] = round(end_ms / 1000 * fps)

            tier_data.append(segment)

        data[tier] = tier_data

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
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .eaf file or a directory containing .eaf files",
    )
    args = parser.parse_args()

    path = args.path.resolve()
    recursive_eaf_to_json(path)


if __name__ == "__main__":
    main()
