import pandas as pd
import json
from pathlib import Path
import argparse


def convert_ocr_json_to_csv(directory: Path, dry_run: bool = False) -> None:
    """
    Recursively find *.ocr.json files, read into DataFrame, and save as .ocr.csv.
    """
    count_total = 0
    count_converted = 0
    count_failed = 0

    for json_file in directory.rglob("*.ocr.json"):
        count_total += 1
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            df = pd.DataFrame(data)

            csv_path = json_file.with_suffix(
                ".csv"
            )  # replaces just the .json file par, so .ocr.csv now
            if dry_run:
                print(
                    f"[DRY RUN] Would convert: {json_file.relative_to(directory)} -> {csv_path.relative_to(directory)}"
                )
            else:
                df.to_csv(csv_path, index=False)
                print(
                    f"Converted: {json_file.relative_to(directory)} -> {csv_path.relative_to(directory)}"
                )
                count_converted += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {json_file}: {e}")
            count_failed += 1

    # Summary
    print("\n=== Conversion Summary ===")
    print(f"Total .ocr.json files found:   {count_total}")
    print(f"Successfully converted:        {count_converted}")
    print(f"Failed conversions:            {count_failed}")
    if dry_run:
        print("No files were written (dry-run mode).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .ocr.json files to .ocr.csv.")
    parser.add_argument("directory", type=Path, help="Directory to search recursively.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing CSV files.",
    )
    args = parser.parse_args()

    convert_ocr_json_to_csv(args.directory, dry_run=args.dry_run)
# cd /data/petabyte/cleong/data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/ocr_json_to_ocr_csv.py .
