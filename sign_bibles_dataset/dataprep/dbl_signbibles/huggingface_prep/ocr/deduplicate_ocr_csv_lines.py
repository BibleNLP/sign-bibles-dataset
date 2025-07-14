import argparse
from pathlib import Path

import pandas as pd


def deduplicate_ocr_csvs(directory: Path, dry_run: bool = False) -> None:
    """
    Recursively find .ocr.csv files and remove consecutive rows with duplicate 'text' fields.
    Writes results to .ocr.filtered.csv files.
    """
    count_total = 0
    count_filtered = 0
    count_failed = 0

    for csv_file in directory.rglob("*.ocr.csv"):
        count_total += 1
        try:
            df = pd.read_csv(csv_file)

            if "text" not in df.columns:
                print(f"[SKIP] No 'text' column in {csv_file}")
                continue

            # Keep first row and any row where 'text' is different from the previous
            filtered_df = df[df["text"].ne(df["text"].shift())]

            output_path = csv_file.with_name(csv_file.stem + ".textchanges.csv")

            if dry_run:
                print(f"[DRY RUN] Would write: {output_path.relative_to(directory)} ({len(filtered_df)} rows)")
            else:
                filtered_df[["frame_index", "text"]].to_csv(output_path, index=False)
                print(f"Filtered: {csv_file.relative_to(directory)} -> {output_path.relative_to(directory)}")
                count_filtered += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {csv_file}: {e}")
            count_failed += 1

    # Summary
    print("\n=== Deduplication Summary ===")
    print(f"Total .ocr.csv files found:    {count_total}")
    print(f"Successfully filtered:         {count_filtered}")
    print(f"Failed to process:             {count_failed}")
    if dry_run:
        print("No files were written (dry-run mode).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate .ocr.csv files by keeping only changed 'text' values.")
    parser.add_argument("directory", type=Path, help="Directory to search recursively.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing output files.",
    )
    args = parser.parse_args()

    deduplicate_ocr_csvs(args.directory, dry_run=args.dry_run)
# cd /data/petabyte/cleong/data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/deduplicate_ocr_csv_lines.py . --dry-run
