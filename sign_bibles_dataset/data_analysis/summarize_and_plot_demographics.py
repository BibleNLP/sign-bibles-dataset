import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_parquet_files(root: Path, filename: str) -> list[pd.DataFrame]:
    dfs = []
    for path in root.rglob(filename):
        try:
            df = pd.read_parquet(path)
            df["source_dir"] = str(path.parent)
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Failed to read {path}: {e}")
    return dfs


def summarize_and_plot_age(age_dfs: list[pd.DataFrame], output_dir: Path):
    if not age_dfs:
        return

    df = pd.concat(age_dfs, ignore_index=True)
    summary = df["age"].describe()
    print("=== Age Summary ===")
    print(summary)

    plt.figure(figsize=(8, 5))
    sns.histplot(df["age"], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png")
    plt.close()


def summarize_and_plot_gender(gender_dfs: list[pd.DataFrame], output_dir: Path):
    if not gender_dfs:
        return

    df = pd.concat(gender_dfs, ignore_index=True)
    counts = df["gender"].value_counts()
    print("\n=== Gender Counts ===")
    print(counts)

    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="gender", order=counts.index)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "gender_distribution.png")
    plt.close()


def summarize_and_plot_race(race_dfs: list[pd.DataFrame], output_dir: Path):
    if not race_dfs:
        return

    df = pd.concat(race_dfs, ignore_index=True)
    counts = df["race"].value_counts()
    print("\n=== Race Counts ===")
    print(counts)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="race", order=counts.index)
    plt.title("Race Distribution")
    plt.xlabel("Race")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "race_distribution.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize DeepFace demographic outputs.")
    parser.add_argument("root_dir", type=str, help="Path to root directory to search.")
    parser.add_argument("--output-dir", type=str, default=".", help="Where to save plots.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    root_path = Path(args.root_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    age_dfs = load_parquet_files(root_path, "age_stats.parquet")
    gender_dfs = load_parquet_files(root_path, "gender_stats.parquet")
    race_dfs = load_parquet_files(root_path, "race_stats.parquet")

    summarize_and_plot_age(age_dfs, output_path)
    summarize_and_plot_gender(gender_dfs, output_path)
    summarize_and_plot_race(race_dfs, output_path)


if __name__ == "__main__":
    main()
