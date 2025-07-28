import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def find_parquet_files(base_dir: Path) -> list[Path]:
    return list(base_dir.rglob("*.parquet"))

def read_and_combine_parquets(parquet_files: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in parquet_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df.drop_duplicates(subset="file_path")

def extract_folder_levels(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    base_dir = base_dir.resolve()
    rel_paths = df["file_path"].apply(lambda p: str(Path(p).resolve().relative_to(base_dir)))
    parts_df = rel_paths.str.split("/", expand=True)
    for level in parts_df.columns:
        df[f"level_{level}"] = parts_df[level]
    return df

def plot_histograms_by_folder_level(df: pd.DataFrame, out_dir: Path):
    duration_col = "duration_sec"
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        if col.startswith("level_"):
            groups = df.groupby(col)[duration_col]
            for group_name, durations in groups:
                if pd.isna(group_name):
                    continue
                plt.figure()
                durations.hist(bins=20)
                plt.title(f"Duration Histogram for {col}={group_name}")
                plt.xlabel("Duration (seconds)")
                plt.ylabel("Count")
                safe_name = f"{col}_{group_name}".replace("/", "_")
                plt.savefig(out_dir / f"{safe_name}.png")
                plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Directory containing .parquet files")
    parser.add_argument("--out_dir", type=Path, default="duration_histograms", help="Directory to save histograms")
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.input_dir)
    if not parquet_files:
        print("No parquet files found.")
        return

    df = read_and_combine_parquets(parquet_files)
    df = extract_folder_levels(df, args.input_dir)
    plot_histograms_by_folder_level(df, args.out_dir)

if __name__ == "__main__":
    main()
