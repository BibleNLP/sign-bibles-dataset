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
        name = filename.split(" in ", 1)[-1].removesuffix(".parquet").strip()
        name = name.replace(" (119 Introductions and Passages)", "")
        name = name.replace(" (119 Introductions and Passages expanded with More Information)", "")
        name = name.replace("Sign Language", "")
        return name.strip()

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


def load_asl_citizen(parquet_path: Path) -> pd.DataFrame:
    """Load ASL Citizen parquet and convert duration_sec -> duration_ms."""
    if not parquet_path.exists():
        logger.error(f"ASL Citizen parquet not found: {parquet_path}")
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    if "duration_sec" not in df.columns:
        logger.error("ASL Citizen parquet missing 'duration_sec' column")
        return pd.DataFrame()

    df = df.copy()
    df["duration_ms"] = df["duration_sec"] * 1000
    df["Language"] = "ASL Citizen"
    # Align with Sign Bible structure: assign a dummy "type" if missing
    if "type" not in df.columns:
        df["type"] = "SIGN"
    logger.info(f"Loaded ASL Citizen parquet with {len(df)} rows")
    return df


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
    """Convert snake_case to 'Title (unit)' format."""
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
    logger.info(f"Creating boxplots. We have {len(df)} rows and languages {df['Language'].unique()}")
    if "type" not in df.columns:
        logger.error("Missing 'type' column in dataframe.")
        return

    for t in df["type"].unique():
        df_t = df[df["type"] == t]

        if df_t.empty:
            continue

        if columns is None:
            columns = list(df_t.select_dtypes(include="number").columns)

        for column in columns:
            df_t_clean = remove_iqr_outliers(df_t, group_col="Language", value_col=column)

            for outliers_removed, df_to_plot in [(False, df_t), (True, df_t_clean)]:
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
                    fig.update_layout(xaxis_tickangle=-45, yaxis_title=pretty_label, title_text=None)

                    base_name = f"{t}_{column}_boxplot_points_{points}_outliersremoved_{outliers_removed}_langs_{len(df['Language'].unique())}"
                    fig.write_html(output_dir / f"{base_name}.html")
                    try:
                        fig.write_image(output_dir / f"{base_name}.pdf")
                    except ValueError as e:
                        logger.warning(f"Could not save PDF: {e}")


def create_bibles_vs_aslcitizen_plot(df: pd.DataFrame, output_dir: Path, column: str, group_sign_languages=True) -> None:
    """Generate combined boxplot comparing Sign Bibles vs ASL Citizen."""
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in dataframe")
        return
    logger.info("Generating All Vs ASL Citizen plot")
    df_clean = remove_iqr_outliers(df, group_col="Language", value_col=column)
    # Collapse all non-ASL Citizen into one group
    if group_sign_languages:
        df_clean = df_clean.copy()
        df_clean["Group"] = df_clean["Language"].apply(
            lambda x: "ASL Citizen" if x == "ASL Citizen" else "All Sign Bibles"
        )
    else:
        df_clean["Group"] = df_clean["Language"]




    fig = px.box(
        df_clean,
        x="Group",
        y=column,
        title=f"All Sign Bibles vs ASL Citizen - {format_label(column)}",
        points=False,
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis_title=None, title_text=None)

    base_name = f"AllSignBibles_vs_ASLCitizen_{column}_boxplot"
    fig.write_html(output_dir / f"{base_name}.html")
    try:
        fig.write_image(output_dir / f"{base_name}.pdf")
        logger.info(f"wrote to {output_dir.resolve()}")
    except ValueError as e:
        logger.warning(f"Could not save PDF: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate boxplots from .parquet files by language and type.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Sign Bible .parquet files",
    )
    parser.add_argument(
        "--asl_citizen",
        type=Path,
        # required=True,
        default="duration_parquets/ASL_Citizen_durations.parquet",
        help="Path to ASL Citizen parquet file",
    )
    parser.add_argument(
        "--columns",
        default="duration_ms",
        type=str,
        help="Comma-separated columns to plot (default: duration_ms)",
    )

    args = parser.parse_args()

    columns = args.columns.split(",") if args.columns else None

    # Load Sign Bible parquets
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        logger.error(f"Invalid input directory: {args.input_dir}")
        return
    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    df_bibles = load_parquet_files(parquet_files)
    create_boxplots(df_bibles, args.input_dir, columns=columns)

    # Load ASL Citizen
    if args.asl_citizen:
        df_asl = load_asl_citizen(args.asl_citizen)

        # Combine
        df_all = pd.concat([df_bibles, df_asl], ignore_index=True)
        if df_all.empty:
            logger.error("No data available after combining.")
            return

        # Generate boxplots
        create_boxplots(df_all, args.input_dir, columns=columns)

        # Special combined comparison
        for col in columns:
            create_bibles_vs_aslcitizen_plot(df_all, args.input_dir, column=col)


if __name__ == "__main__":
    main()
