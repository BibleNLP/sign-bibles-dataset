#!/usr/bin/env python3
"""Find unique verse indices from transcript JSON files grouped by language."""

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
    parser.add_argument("--depth", type=int, default=2, help="Depth of subfolders to search.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save output Parquet files and LaTeX. Default: ./unique_verses/depth_{depth}",
    )
    parser.add_argument(
        "--eng_ebible",
        type=Path,
        default="/opt/home/cleong/data/eBible/ebible/corpus/eng-engbsb.txt",
        help="Path to an English translation from the eBible corpus",
    )

    parser.add_argument(
        "--eng_vref",
        type=Path,
        default="/opt/home/cleong/data/eBible/ebible/metadata/vref.txt",
        help="Path to vref.txt from the eBible corpus",
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
    return {lang: {verse for verses in group for verse in verses} for lang, group in grouped}


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


def load_english_verse_indices(path: Path) -> set[int]:
    """
    Load verse indices from an English eBible file.
    Each line corresponds to a verse; non-empty lines are counted.
    """
    with path.open(encoding="utf-8") as f:
        return {i for i, line in enumerate(f) if line.strip()}


def load_vrefs(path: Path):
    """
    Load verse indices from an English eBible file.
    Each line corresponds to a verse; non-empty lines are counted.
    """
    vref_dict = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            vref_dict[i] = line.strip()
    return vref_dict


def plot_supervenn_diagram(
    verse_counts: dict[str, set[int]],
    output_file: Path | None = None,
    combine_identical: bool = True,
) -> None:
    """
    Plot a Supervenn diagram showing the overlap of verses between languages.

    Args:
        verse_counts (dict[str, set[int]]): Mapping from language to a set of verse IDs.
        output_file (Path | None): If provided, the plot will be saved to this file.
        combine_identical (bool): If True, combine labels that have identical verse sets.

    """
    if combine_identical:
        grouped: dict[frozenset[int], list[str]] = defaultdict(list)
        for label, verse_set in verse_counts.items():
            grouped[frozenset(verse_set)].append(label)

        sets = [set(verse_set) for verse_set in grouped]
        labels = [", ".join(group) for group in grouped.values()]
    else:
        sets = list(verse_counts.values())
        labels = list(verse_counts.keys())

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 26,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    height_per_set = 0.8
    width = max(12, len(sets) * 0.8)
    height = max(8, len(sets) * height_per_set)
    plt.figure(figsize=(width, height))

    supervenn_obj = supervenn(
        sets,
        labels,
        side_plots=False,
        min_width_for_annotation=100_000,
        rotate_col_annotations=True,
    )

    supervenn_obj.axes["main"].set_xlabel("")
    supervenn_obj.axes["main"].set_xticks([])
    supervenn_obj.axes["main"].grid(False)

    for label in supervenn_obj.axes["main"].get_xticklabels():
        label.set_visible(False)

    plt.title("Verse Overlaps Across Languages", fontsize=24)
    supervenn_obj.figure.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        logging.info("Saved supervenn to %s", output_file.resolve())

    plt.close()


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

    english_indices = load_english_verse_indices(args.eng_ebible)
    vref_dict = load_vrefs(args.eng_vref)
    plot_supervenn_diagram(verse_counts, output_file=output_dir / "verse_overlap_supervenn_no_text.png")
    verse_counts["English Text (BSB)"] = english_indices
    logging.info(f"Loaded {len(english_indices)} nonblank verse indices from {args.eng_ebible}")

    plot_supervenn_diagram(verse_counts, output_file=output_dir / "verse_overlap_supervenn.png")

    all_languages_set = set().union(*(s for k, s in verse_counts.items() if k != "English Text (BSB)"))
    verse_counts["All Sign Languages"] = all_languages_set

    for lang, verses in verse_counts.items():
        if lang == "English":
            continue
        pair_counts = {
            "English": english_indices,
            lang: verses,
        }
        plot_supervenn_diagram(
            pair_counts, output_file=output_dir / f"supervenn_{lang.lower().replace(' ', '_')}_vs_english.png"
        )
        missing_verses = verses - english_indices
        print(f"{lang} has these that AREN'T in English\n")
        for missing_verse_index in missing_verses:
            logging.info(vref_dict.get(missing_verse_index, missing_verse_index))

        rows = []
        for missing_verse_index in sorted(missing_verses):
            rows.append(
                {
                    "missing_index": missing_verse_index,
                    "Verse": vref_dict.get(missing_verse_index, ""),
                }
            )
        out_path = output_dir / f"{lang.lower()}_missing_verses.csv"
        missing_verses_df = pd.DataFrame(rows, columns=["missing_index", "Verse"])
        missing_verses_df.to_csv(out_path, index=False)
        logging.info("Saved missing verses CSV to %s", out_path)

        logging.info(f"Verses in {lang} not in English:\n{missing_verses_df}")

    logging.info("Results written to: %s", output_dir)


if __name__ == "__main__":
    main()
