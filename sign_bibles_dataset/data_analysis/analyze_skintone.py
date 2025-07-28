import argparse
import logging
from pathlib import Path

import pandas as pd


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=level)


def find_result_csvs(root: Path) -> list[Path]:
    """Recursively find all skintone/result.csv files under the given root."""
    return list(root.rglob("skintone/result.csv"))


def read_clean_csv(path: Path) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Read a CSV file, skipping bad lines and returning both the DataFrame and the list of skipped rows."""
    valid_rows = []
    bad_rows = []

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for lineno, line in enumerate(f, start=2):
            row = line.strip().split(",")
            if len(row) == len(header):
                valid_rows.append(row)
            else:
                bad_rows.append((lineno, line.strip()))

    df = pd.DataFrame(valid_rows, columns=header)
    return df, bad_rows


def collect_skintone_results(root: Path) -> pd.DataFrame:
    all_dfs = []
    total_bad_rows = 0

    for csv_path in find_result_csvs(root):
        logging.info(f"Reading: {csv_path}")
        df, bad_rows = read_clean_csv(csv_path)
        if bad_rows:
            logging.warning(f"{csv_path} had {len(bad_rows)} malformed rows (e.g. line {bad_rows[0][0]})")
            total_bad_rows += len(bad_rows)

        df["source_file"] = str(csv_path)  # Add source file info
        all_dfs.append(df)

    if not all_dfs:
        logging.warning("No valid result.csv files found.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined DataFrame has {len(combined_df)} rows.")
    if total_bad_rows > 0:
        logging.info(f"Total malformed rows skipped: {total_bad_rows}")
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Collect skintone result.csv files recursively.")
    parser.add_argument("root", type=Path, help="Root folder to search")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    combined_df = collect_skintone_results(args.root)
    print(combined_df.describe())


if __name__ == "__main__":
    main()
