import argparse
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_language_from_filename(filename: str) -> str:
    """Extract the language name from the filename."""
    if " in " in filename:
        # e.g. Chronological Bible Translation in American Sign Language (119 Introductions and Passages expanded with More Information)

        # get everything after "in"
        name = filename.split(" in ", 1)[-1].removesuffix(".parquet").strip()

        # "Sign Language" is redundant
        name = name.replace(" (119 Introductions and Passages)", "")
        name = name.replace(" (119 Introductions and Passages expanded with More Information)", "")
        name = name.replace("Sign Language", "")
        # name = name.replace(" ", "")
        # name = " ".join(name.split())
        name = name.strip()
        return name

    return "Unknown"


def load_parquet_files(parquet_files: list[Path]) -> pd.DataFrame:
    """Load all .parquet files and add language column."""
    all_dfs = []
    for file_path in parquet_files:
        if not file_path.is_file():
            logger.warning(f"Skipping non-file path: {file_path}")
            continue
        if file_path.suffix != ".parquet":
            logger.warning(f"Skipping non-parquet file: {file_path}")
            continue

        df = pd.read_parquet(file_path)
        language = extract_language_from_filename(file_path.stem)
        df["Language"] = language
        all_dfs.append(df)
        logger.info(f"Loaded: {file_path.name} with language '{language}'")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def remove_iqr_outliers(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Remove IQR outliers per group (e.g., per language)."""

    def _filter_group(group):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return group[(group[value_col] >= lower_bound) & (group[value_col] <= upper_bound)]

    return df.groupby(group_col, group_keys=False).apply(_filter_group)


def format_label(column_name: str) -> str:
    """Convert snake_case to 'Title (unit)' format, e.g., duration_ms -> Duration (ms)"""
    suffix_units = {
        "ms": "(ms)",
        "s": "(s)",
        "sec": "(s)",
        "seconds": "(s)",
        "hz": "(Hz)",
        "fps": "(fps)",
        "px": "(px)",
        "count": "(count)",
        "frames": "(frames)",
        "score": "(score)",
        "ratio": "(ratio)",
        "percent": "(%)",
    }

    parts = column_name.lower().split("_")
    if not parts:
        return column_name

    if parts[-1] in suffix_units:
        unit = suffix_units[parts[-1]]
        parts = parts[:-1]
    else:
        unit = ""

    title = " ".join(p.capitalize() for p in parts)
    return f"{title} {unit}".strip()


def create_boxplots(df: pd.DataFrame, output_dir: Path, columns=None) -> None:
    """Create and save box plots for each unique 'type'."""
    if "type" not in df.columns:
        logger.error("Missing 'type' column in dataframe.")
        return

    for t in df["type"].unique():  # SENTENCE and SIGN
        df_t = df[df["type"] == t]

        if df_t.empty:
            logger.warning(f"No data found for type '{t}'")
            continue

        # Plot (auto-detects numeric columns)
        if columns is None:
            columns = list(df_t.select_dtypes(include="number").columns)

        for column in columns:
            df_t_clean = remove_iqr_outliers(df_t, group_col="Language", value_col=column)

            removed_outliers_values = [True, False]

            for outliers_removed in removed_outliers_values:
                if outliers_removed:
                    df_to_plot = df_t_clean
                else:
                    df_to_plot = df_t

                pretty_label = format_label(column)
                title = f"Autosegmenter {t} - {pretty_label}"

                for points in ["suspectedoutliers", False]:
                    fig = px.box(
                        df_to_plot,
                        x="Language",
                        y=column,
                        title=title,
                        points=points,
                    )
                    fig.update_layout(xaxis_tickangle=-45, yaxis_title=pretty_label)
                    fig.update_layout(title_text=None)
                    

                    # Output paths
                    base_name = f"{t}_{column}_boxplot_points_{points}_outliersremoved_{outliers_removed}"
                    html_path = output_dir / f"{base_name}.html"
                    pdf_path = output_dir / f"{base_name}.pdf"

                    

                    fig.write_html(html_path)
                    try:
                        fig.write_image(pdf_path)
                        logger.info(f"Saved plot to: {html_path} and {pdf_path}")
                    except ValueError as e:
                        logger.warning(f"Could not save PDF (missing kaleido?): {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate boxplots from .parquet files by language and type.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("./autosegmentation_analysis/depth_2/model_E4s-1/"),
        help="Directory containing .parquet files",
    )
    parser.add_argument(
        "--columns",
        default="duration_ms",
        type=str,
        help="Comma-separated: Which columns to plot? default: duration_ms",
    )

    args = parser.parse_args()

    if args.columns is not None:
        columns = args.columns.split(",")
    else:
        columns = None

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Invalid input directory: {input_dir}")
        return

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No .parquet files found in: {input_dir}")
        return

    logger.info(f"Found {len(parquet_files)} .parquet files")

    df = load_parquet_files(parquet_files)
    if df.empty:
        logger.warning("No data loaded from parquet files.")
        return

    create_boxplots(df, input_dir, columns=columns)


if __name__ == "__main__":
    main()
