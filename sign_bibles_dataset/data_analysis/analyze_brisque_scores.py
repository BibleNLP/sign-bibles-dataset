#!/usr/bin/env python3
"""
Aggregate BRISQUE score parquet files per subfolder
(at configurable depth), produce summary statistics,
and export boxplots.
"""

# TODO: save aggregated dfs as parquets to the out folder
# TODO: improved names like in generate_autosegmenter_boxplots

import argparse
import json
import logging
from collections import deque
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate BRISQUE score parquet files and generate reports.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the root folder containing subfolders with parquet files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where reports and plots will be saved.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Depth of subfolders to aggregate (1 = immediate children, 2 = grandchildren, etc.).",
    )

    parser.add_argument(
        "--group-by-language",
        action="store_true",
        help="If true, group by language, not subfolder",
    )
    return parser.parse_args()


def find_target_subfolders(root: Path, depth: int) -> list[Path]:
    """Return all subfolders at exactly the given depth under root, efficiently."""
    if depth < 1:
        raise ValueError("Depth must be >= 1")

    results = []
    queue = deque([(root, 0)])  # (path, current_depth)

    while queue:
        current, d = queue.popleft()
        if d == depth:
            results.append(current)
            continue
        if d < depth:
            # only descend if not yet at depth
            try:
                for child in current.iterdir():
                    if child.is_dir():
                        queue.append((child, d + 1))
            except PermissionError:
                continue

    return sorted(results)


def find_parquet_files(subfolder: Path) -> list[Path]:
    return list(subfolder.rglob("*.brisque_scores.parquet"))


def load_scores(parquet_files: list[Path]) -> pd.DataFrame:
    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["score"])
            language = extract_language_from_filename(f)
            logging.debug(f"Language for {f}: {language}")
            df["Language"] = language
            df["source"] = str(f)
            dfs.append(df)
        except (ValueError, OSError, KeyError) as e:
            logging.warning("Skipping %s: %s", f, e)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["score"])


def summarize_scores(df: pd.DataFrame) -> dict[str, float | str | int]:
    if df.empty:
        return {}

    min_idx = df["score"].idxmin()
    max_idx = df["score"].idxmax()

    return {
        "count": int(df["score"].count()),
        "mean": float(df["score"].mean()),
        "std": float(df["score"].std()),
        "min": float(df["score"].min()),
        "min_source": str(df.loc[min_idx, "source"]) if pd.notna(min_idx) else "",
        "25%": float(df["score"].quantile(0.25)),
        "50%": float(df["score"].median()),
        "75%": float(df["score"].quantile(0.75)),
        "max": float(df["score"].max()),
        "max_source": str(df.loc[max_idx, "source"]) if pd.notna(max_idx) else "",
    }


def save_summary_json(summary: dict[str, dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_boxplot(df: pd.DataFrame, output_prefix: Path, category: str | None = None) -> None:
    logging.info(f"Generating boxplot for category {category}, languages: {df['Language'].unique()}")
    logging.info(df.head())
    if df.empty:
        return
    if category is None:
        fig = px.box(df, y="score")
    else:
        fig = px.box(df, x=category, y="score")
    fig.update_layout(
        title=None,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    fig.write_html(str(output_prefix.with_suffix(".html")))
    fig.write_image(str(output_prefix.with_suffix(".pdf")))


def extract_language_from_filename(filename: str) -> str:
    """Extract the human-readable language name from a BRISQUE score file path."""
    path = Path(filename)

    for parent in path.parents:
        if "sign language" in parent.name.lower():
            # get everything after "in"
            name = parent.name
            name = name.split(" in ", 1)[-1].removesuffix(".parquet").strip()

            # Clean redundant text
            name = name.replace(" (119 Introductions and Passages)", "")
            name = name.replace(" (119 Introductions and Passages expanded with More Information)", "")
            name = name.replace("Sign Language", "").strip()
            return name

    return "Unknown"


def save_summary_csv_and_tex(summary: dict[str, dict[str, float | str | int]], output_dir: Path) -> None:
    if not summary:
        logging.warning("No summary data to save.")
        return

    df = pd.DataFrame.from_dict(summary, orient="index")
    df.index.name = "subfolder"

    csv_path = output_dir / "summary.csv"
    tex_path = output_dir / "summary.tex"

    df.to_csv(csv_path)
    logging.info("Saved CSV summary to %s", csv_path)

    df.to_latex(
        tex_path,
        index=True,
        float_format="%.2f",
        bold_rows=False,
        na_rep="--",
        column_format="l" + "r" * (len(df.columns)),  # align: left index + right for each column
        escape=True,
        caption="Summary of BRISQUE scores by subfolder.",
        label="tab:brisque_summary",
        longtable=False,
        multicolumn=False,
        multicolumn_format="c",
        position="htbp",
    )
    logging.info("Saved LaTeX summary to %s", tex_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input_dir.is_dir():
        logging.error("Input directory does not exist: %s", args.input_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_summary = {}
    combined_dfs = []

    # Find subfolders at the requested depth
    subfolders = find_target_subfolders(args.input_dir, args.depth)

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        label = "/".join(subfolder.relative_to(args.input_dir).parts)
        logging.info("Processing subfolder: %s", label)

        parquet_files = find_parquet_files(subfolder)
        if not parquet_files:
            logging.info("No parquet files found in %s", subfolder)
            continue
        output_prefix = args.output_dir / f"{label.replace('/', '_')}_scores"
        df = load_scores(parquet_files)
        logging.info(f"Languages in folder {df['Language'].unique()}")
        if not df.empty:
            df["subfolder"] = label
            combined_dfs.append(df)
            df.to_parquet(f"{output_prefix}.parquet")

        summary = summarize_scores(df)
        overall_summary[label] = summary

        # Save per-subfolder plots

        save_boxplot(df, output_prefix)

    # Save JSON summary
    save_summary_json(overall_summary, args.output_dir / "summary.json")

    # Combined boxplot
    if combined_dfs:
        all_df = pd.concat(combined_dfs, ignore_index=True)
        all_df.to_parquet(args.output_dir / "all_brisque_scores.parquet")

        if args.group_by_language:
            category = "Language"
        else:
            category = "subfolder"

        save_boxplot(all_df, args.output_dir / "all_subfolders_scores", category=category)

    logging.info("Done. Reports written to %s", args.output_dir)


if __name__ == "__main__":
    main()
