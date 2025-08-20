import argparse
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


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

    for csv_path in tqdm(find_result_csvs(root), desc="reading result.csv files"):
        logging.debug(f"Reading: {csv_path}")
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


def plot_skintone_distribution(df: pd.DataFrame, output_dir: Path, tight_layout: bool = False, by_file: bool = False):
    """Plot and save the distribution of skin tones."""
    if "skin tone" not in df.columns:
        logging.error("'skin tone' column not found.")
        return

    if by_file:
        # Count unique source files per skin tone
        grouped = df.groupby(["skin tone", "source_file"]).size().reset_index(name="dummy")
        tone_counts = grouped.groupby("skin tone")["source_file"].nunique().reset_index(name="count")
        y_axis_title = "Unique File Count"
        suffix = "_by_file"
    else:
        # Frame-based count
        tone_counts = df["skin tone"].value_counts().reset_index()
        tone_counts.columns = ["skin tone", "count"]
        y_axis_title = "Frame Count"
        suffix = ""

    tone_counts = tone_counts.sort_values("skin tone")  # optional: sort by label

    fig = px.bar(
        tone_counts,
        x="skin tone",
        y="count",
        title="Skin Tone Distribution" if not tight_layout else None,
        text="count",
        color="skin tone",
        color_discrete_map={tone: tone for tone in tone_counts["skin tone"]},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Skin Tone",
        yaxis_title=y_axis_title,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        margin=dict(l=40, r=20, t=40 if not tight_layout else 5, b=40),
        showlegend=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_base_name = f"skin_tone_distribution{suffix}"
    if tight_layout:
        out_base_name += "_tight"

    fig.write_html(output_dir / f"{out_base_name}.html")
    try:
        fig.write_image(output_dir / f"{out_base_name}.pdf")
        logging.info(f"Saved plot to: {out_base_name}.html and .pdf")
    except ValueError as e:
        logging.warning(f"Could not save PDF (is kaleido installed?): {e}")


def main():
    parser = argparse.ArgumentParser(description="Collect skintone result.csv files recursively.")
    parser.add_argument("root", type=Path, help="Root folder to search")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", type=Path, default=Path("skintone_plots"), help="Output directory for plots")
    parser.add_argument("--tight", action="store_true", help="Enable tight layout for PDF plots (no titles)")
    parser.add_argument(
        "--by-file", action="store_true", help="Count unique source files per skin tone instead of total frames"
    )

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    combined_df = collect_skintone_results(args.root)
    output_dir = args.output_dir 
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(args.output_dir / "combined_skintone_results.parquet")
    combined_df = combined_df[combined_df["face id"] != "NA"]

    print("After filtering for NA:")
    print(combined_df.describe())

    plot_skintone_distribution(combined_df, output_dir=output_dir, tight_layout=args.tight, by_file=args.by_file)
    


if __name__ == "__main__":
    main()
