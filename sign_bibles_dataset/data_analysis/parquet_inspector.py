import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Inspect a Parquet file using pandas.")

    parser.add_argument("file", type=Path, help="Path to the .parquet file")
    parser.add_argument(
        "--cols", type=str, help="Comma-separated column list. If given, will restrict the columns to these"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--head", action="store_true", help="Show the first few rows")
    group.add_argument("--tail", action="store_true", help="Show the last few rows")
    group.add_argument("--info", action="store_true", help="Show DataFrame info")
    group.add_argument("--describe", action="store_true", help="Show summary statistics")
    group.add_argument("--sum", action="store_true", help="Find sum of all numeric columns")
    group.add_argument("--mean", action="store_true", help="Find mean of all numeric columns")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
    else:
        print(args.file.name)

    df = pd.read_parquet(args.file)

    if args.cols:
        cols = args.cols.split(",")
        # print(f"Selecting columns {cols}")
        df = df[args.cols.split(",")]

    if args.head:
        print("head:")
        print(df.head())

    if args.tail:
        print("tail:")
        print(df.tail())

    if args.info:
        df.info()

    if args.describe:
        print(df.describe())

    if args.mean:
        for col in df.select_dtypes(include="number").columns:
            print(f"{col} mean: {df[col].mean()}")

    if args.sum:
        # print("sums:")
        for col in df.select_dtypes(include="number").columns:
            print(f"{col} sum: {df[col].sum()}")


if __name__ == "__main__":
    main()
