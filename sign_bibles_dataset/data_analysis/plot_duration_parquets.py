import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_parquet_files(base_dir: Path) -> list[Path]:
    return list(base_dir.rglob("*.parquet"))


def read_and_combine_parquets(parquet_files: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in parquet_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["file_path"] = combined_df["file_path"].str.replace(
        "/data/petabyte/cleong/data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads/", "top/"
    )
    combined_df = combined_df[~combined_df["file_path"].str.contains("cleong")]
    combined_df = combined_df[~combined_df["file_path"].str.contains("segments")]

    print(combined_df.sample(10))
    # exit()
    return combined_df.drop_duplicates(subset="file_path")


def extract_folder_levels_until_convergence(df: pd.DataFrame) -> pd.DataFrame:
    path_parts = df["file_path"].apply(lambda p: Path(p).resolve().parts)
    max_depth = max(len(parts) for parts in path_parts)

    for i in range(-1, -max_depth - 1, -1):
        col = f"level_{abs(i)}"
        df[col] = path_parts.apply(lambda parts: parts[i] if len(parts) >= abs(i) else None)

        # Stop when all files fall into the same group
        if df[col].nunique(dropna=True) <= 1:
            break

    return df


def plot_histograms_by_folder_level(df: pd.DataFrame, out_dir: Path, min_group_size: int = 5):
    duration_col = "duration_sec"
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        if not col.startswith("level_"):
            continue

        groups = df.groupby(col)[duration_col]
        for group_name, durations in groups:
            if pd.isna(group_name) or len(durations) < min_group_size:
                continue

            plt.figure()
            # durations.hist(bins=30)
            sns.histplot(durations, bins=30, kde=True)
            # plt.title(f"Duration Histogram for {col}={group_name} ({len(durations)} videos)")
            plt.xlabel("Duration (seconds)")
            plt.ylabel("Count")
            safe_name = f"{col}_{group_name}".replace("/", "_").replace(" ", "_")
            plt.tight_layout()
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
    df = extract_folder_levels_until_convergence(df)
    plot_histograms_by_folder_level(df, args.out_dir)


if __name__ == "__main__":
    main()
