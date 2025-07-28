import argparse
import json
import logging
from pathlib import Path

import pandas as pd


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger("segment_parser")


def parse_segments(file_path: Path, logger: logging.Logger) -> list[dict]:
    segments = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return segments

    for category in ("SENTENCE", "SIGN"):
        seg_list = data.get(category, [])
        for idx, seg in enumerate(seg_list):
            try:
                start_ms = seg["start_ms"]
                end_ms = seg["end_ms"]
                duration = end_ms - start_ms
                segments.append(
                    {
                        "seg_idx": idx,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": duration,
                        "file": str(file_path),
                        "type": category,
                    }
                )
            except KeyError as e:
                logger.warning(f"Missing expected key in {file_path}: {e}")
    return segments


def main():
    parser = argparse.ArgumentParser(description="Parse autosegmented JSON segments.")
    parser.add_argument("dir", type=Path, help="Directory to recursively search for JSONs")
    parser.add_argument("model", type=str, help="Model name used in JSON filenames")
    parser.add_argument("output", type=Path, help="Path to output Parquet file (.parquet)")

    args = parser.parse_args()

    log = setup_logger()

    if not args.dir.is_dir():
        log.error(f"Provided path is not a directory: {args.dir}")
        return

    if not args.output.suffix == ".parquet":
        log.error("Output path must end with .parquet")
        return

    pattern = f"*.{args.model}.autosegmented_segments.json"
    json_files = list(args.dir.rglob(pattern))
    log.info(f"Found {len(json_files)} matching files for model '{args.model}'")

    all_segments = []
    for json_file in json_files:
        segments = parse_segments(json_file, log)
        all_segments.extend(segments)

    if not all_segments:
        log.warning("No segments parsed.")
        return

    df = pd.DataFrame(all_segments)
    log.info(f"Parsed {len(df)} segments. Writing to {args.output}")

    df.to_parquet(args.output, index=False)
    log.info(df.describe())
    log.info("Done.")


if __name__ == "__main__":
    main()
