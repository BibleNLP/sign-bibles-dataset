#!/usr/bin/env python3
"""
Find unique verse indices from transcript JSON files grouped by language.
"""

import argparse
import json
import logging
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from supervenn import supervenn
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate unique verse indices grouped by language.")
    parser.add_argument("input_dir", type=Path, help="Root directory to search for transcripts.")
    parser.add_argument("--depth", type=int, default=3, help="Depth of subfolders to search.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save output Parquet files and LaTeX. Default: ./unique_verses/depth_{depth}",
    )
    parser.add_argument(
        "--ebible_bsb",
        type=Path,
        default="/opt/home/cleong/data/eBible/ebible/corpus/eng-engbsb.txt",
        help="Directory to save output Parquet files and LaTeX. Default: ./unique_verses/depth_{depth}",
    )

    return parser.parse_args()


def find_target_subfolders(root: Path, depth: int) -> list[Path]:
    """Return all subfolders at exactly the given depth under root."""
    if depth < 1:
        raise ValueError("Depth must be >= 1")

    results = []
    queue = deque([(root, 0)])

    while queue:
        current, d = queue.popleft()
        if d == depth:
            results.append(current)
            continue
        try:
            for child in current.iterdir():
                if child.is_dir():
                    queue.append((child, d + 1))
        except PermissionError:
            continue

    return sorted(results)


def extract_language_name(path: Path, root: Path) -> str:
    """
    Extract human-readable language name from transcript file path.
    Looks for a parent directory with 'sign language' in its name,
    and pulls the string after 'in' as the name.
    """
    try:
        relative_parts = path.relative_to(root).parents
    except ValueError:
        relative_parts = path.parents

    for parent in relative_parts:
        parent_name = parent.name.replace("_", " ").lower()
        if "sign language" in parent_name:
            name = parent_name
            name = name.split(" in ", 1)[-1].strip()
            name = name.split("sign")[0].strip()

            name = name.replace("Sign Language", "")
            name = name.replace(" (119 Introductions and Passages)", "")
            name = name.replace(" (119 Introductions and Passages expanded with More Information)", "")
            name = " ".join([s.capitalize() for s in name.split()])
            name = name.replace("Delhi", "(Delhi)")
            return name.strip()

    # fallback: use 3-letter code
    for part in path.parts:
        if part.isalpha() and len(part) == 3:
            return part
    return "Unknown"


def process_transcript_files(files: list[Path], root: Path) -> pd.DataFrame:
    records = []

    for f in files:
        try:
            with f.open(encoding="utf-8") as fp:
                transcripts = json.load(fp)

            if not isinstance(transcripts, list):
                logging.warning("Skipping %s: not a list of transcripts", f)
                continue

            language_name = extract_language_name(f, root)

            for i, segment in enumerate(transcripts):
                verse_indices = segment.get("biblenlp-vref", [])
                if isinstance(verse_indices, list):
                    records.append(
                        {
                            "path": str(f),
                            "segment_index": i,
                            "verse_indices": verse_indices,
                            "Language": language_name,
                        }
                    )

        except (OSError, json.JSONDecodeError, TypeError) as e:
            logging.warning("Skipping %s: %s", f, e)
    df = pd.DataFrame.from_records(records)
    logging.info(f"Languages: {df['Language'].unique()}")
    return df


def save_language_dataframe(df: pd.DataFrame, output_dir: Path) -> dict[str, set[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "all_languages.parquet", index=False)
    logging.info("Saved combined DataFrame with %d rows", len(df))

    grouped = df.groupby("Language")["verse_indices"]
    return {lang: set(verse for verses in group for verse in verses) for lang, group in grouped}


def save_latex_table(verse_counts: dict[str, set[int]], output_file: Path) -> None:
    all_verses = set().union(*verse_counts.values())
    verse_counts_with_total = dict(verse_counts)
    verse_counts_with_total["All languages"] = all_verses

    sorted_counts = sorted(
        verse_counts_with_total.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    with output_file.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{|l|r|}\n\\hline\n")
        f.write("Language & Unique Verses \\\\\n\\hline\n")
        for lang, verses in sorted_counts:
            f.write(f"{lang} & {len(verses)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


import matplotlib.pyplot as plt
from pathlib import Path
from supervenn import supervenn
import logging


def plot_supervenn_diagram(verse_counts: dict[str, set[int]], output_file: Path | None = None) -> None:
    """
    Plot a Supervenn diagram showing the overlap of verses between languages.

    Args:
        verse_counts (dict[str, set[int]]): Mapping from language to a set of verse IDs.
        output_file (Path | None): If provided, the plot will be saved to this file.
    """
    sets = list(verse_counts.values())
    labels = list(verse_counts.keys())

    # Configure global font size
    plt.rcParams.update(
        {
            "font.size": 20,  # main font size
            "axes.titlesize": 24,  # title font size
            "axes.labelsize": 20,  # axis label font size
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    # Choose a manageable figure size, or scale up for many labels
    height_per_set = 0.6
    width = max(12, len(sets) * 0.8)
    height = max(8, len(sets) * height_per_set)
    plt.figure(figsize=(width, height))

    supervenn(
        sets,
        labels,
        side_plots=False,
        widths_minmax_ratio=0.05,
        min_width_for_annotation=100,
    )

    plt.title("Verse Overlaps Across Languages", fontsize=24)
    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        logging.info(f"Saved supervenn to {output_file.resolve()}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output_dir or Path(f"./unique_verses/depth_{args.depth}")
    subfolders = find_target_subfolders(args.input_dir, args.depth)

    all_records = []

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        label = "/".join(subfolder.relative_to(args.input_dir).parts)
        logging.info("Subfolder: %s", subfolder)

        transcript_files = list(subfolder.rglob("*transcripts.json"))
        if not transcript_files:
            continue

        df = process_transcript_files(transcript_files, args.input_dir)
        if not df.empty:
            df["dataset_label"] = label
            all_records.append(df)

    if not all_records:
        logging.warning("No valid transcript files found.")
        return

    full_df = pd.concat(all_records, ignore_index=True)
    logging.info("Processed %d rows across %d datasets/languages", len(full_df), len(all_records))

    verse_counts = save_language_dataframe(full_df, output_dir)
    save_latex_table(verse_counts, output_dir / "verse_counts.tex")

    plot_supervenn_diagram(verse_counts, output_file=output_dir / "verse_overlap_supervenn.png")

    logging.info("Results written to: %s", output_dir)


if __name__ == "__main__":
    main()
