import argparse
import logging
from pathlib import Path

import pandas as pd

from sign_bibles_dataset.data_analysis.plot_histograms import plot_histogram_df

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("histogram_by_type")


def plot_histograms_by_type(parquet_path: Path, column: str) -> None:
    try:
        df = pd.read_parquet(parquet_path)

        if "type" not in df.columns:
            log.warning(f"'type' column missing in {parquet_path.name}, skipping.")
            return

        unique_types = df["type"].dropna().unique()
        if len(unique_types) == 0:
            log.warning(f"No non-null 'type' values in {parquet_path.name}, skipping.")
            return

        for type_value in unique_types:
            df_subset = df[df["type"] == type_value]
            if df_subset.empty:
                continue

            output_stem = parquet_path.with_suffix("").with_name(f"{parquet_path.stem}_{type_value}")
            plot_histogram_df(df_subset, column, output_stem)

    except Exception as e:
        log.error(f"Failed to process {parquet_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot histograms by 'type' from parquet files.")
    parser.add_argument("directory", type=Path, help="Directory to search for parquet files.")
    parser.add_argument("column", type=str, help="Column to plot histogram for.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        log.error(f"{args.directory} is not a valid directory.")
        return

    parquet_files = list(args.directory.rglob("*.parquet"))
    if not parquet_files:
        log.warning("No parquet files found.")
        return

    for parquet_path in parquet_files:
        log.info(f"Processing {parquet_path}")
        plot_histograms_by_type(parquet_path, args.column)


if __name__ == "__main__":
    main()
