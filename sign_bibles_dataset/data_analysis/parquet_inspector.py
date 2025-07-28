import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Inspect a Parquet file using pandas.")

    parser.add_argument("file", type=Path, help="Path to the .parquet file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--head", action="store_true", help="Show the first few rows")
    group.add_argument("--tail", action="store_true", help="Show the last few rows")
    group.add_argument("--info", action="store_true", help="Show DataFrame info")
    group.add_argument("--describe", action="store_true", help="Show summary statistics")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.file)

    if args.head:
        print(df.head())
    elif args.tail:
        print(df.tail())
    elif args.info:
        df.info()
    elif args.describe:
        print(df.describe())


if __name__ == "__main__":
    main()
