import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("histogram_plotter")


def plot_histogram_df(
    df: pd.DataFrame,
    column: str,
    output_stem: Path,
    clip_quantile: float = 0.995,
    log_scale: bool = False,
    use_percent: bool = False,
) -> None:
    import matplotlib as mpl

    if column not in df.columns:
        log.warning(f"Column '{column}' not found in DataFrame, skipping.")
        return

    values = df[column].dropna()
    if clip_quantile is not None:
        upper_limit = values.quantile(clip_quantile)
        values = values[values <= upper_limit]

    # Theme for colorblind-friendly and clean design
    sns.set_context("paper", font_scale=1.4)  # Big enough fonts
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # Use serif font (like Times) to match LaTeX
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["pdf.fonttype"] = 42  # Embed fonts in PDF (vector)
    mpl.rcParams["axes.edgecolor"] = "none"  # No border lines

    plt.figure()
    sns.histplot(values, kde=False)

    if log_scale:
        plt.xscale("log")

    # Capitalized labels, no jargon
    label = column.replace("_", " ").replace("-", " ").title()
    plt.xlabel(label)
    plt.ylabel("Count")

    if use_percent:
        plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=len(values)))

    plt.tight_layout()

    # Save with sanitized file name (remove spaces for arXiv)
    safe_name = output_stem.name.replace(" ", "")
    png_path = output_stem.with_name(f"{safe_name}_{column}_histogram.png")
    pdf_path = output_stem.with_name(f"{safe_name}_{column}_histogram.pdf")

    for path in [png_path, pdf_path]:
        plt.savefig(path, bbox_inches="tight", dpi=300)  # Trim whitespace, high res
        log.info(f"Saved: {path}")

    plt.close()


def plot_histogram_file(parquet_path: Path, column: str) -> None:
    try:
        if parquet_path.name.endswith(".csv"):
            df = pd.read_csv(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)
        output_stem = parquet_path.with_suffix("")  # e.g., foo.parquet -> foo
        plot_histogram_df(df, column, output_stem)
        logging.info(df.describe())
    except Exception as e:
        log.error(f"Failed to process {parquet_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot histograms from parquet files.")
    parser.add_argument("directory", type=Path, help="Directory to search for parquet files.")
    parser.add_argument("column", type=str, help="Column to plot histogram for.")
    args = parser.parse_args()

    # if not args.directory.is_dir():
    #     log.error(f"{args.directory} is not a valid directory.")
    #     return

    if args.directory.is_file():
        parquet_files = [args.directory]
    else:
        parquet_files = list(args.directory.rglob("*.parquet"))
    if not parquet_files:
        log.warning("No parquet files found.")
        return

    for parquet_path in parquet_files:
        log.info(f"Processing {parquet_path}")
        plot_histogram_file(parquet_path, args.column)


if __name__ == "__main__":
    main()
