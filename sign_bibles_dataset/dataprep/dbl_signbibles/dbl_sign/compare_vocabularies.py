#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def get_csv_vocab(csv_path: Path, column: str) -> set:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}. Available columns: {list(df.columns)}")
    return set(df[column].dropna().astype(str).str.upper().unique())


def tokenize(text: str) -> set:
    words = text.upper().split()
    return set(word.strip('.,;:!?"“”‘’()[]{}') for word in words if word.strip())


def compare_and_write(text_path: Path, vocab: set, start: int, end: int, out_path: Path = None):
    with open(text_path, encoding="utf-8") as f:
        lines = f.readlines()

    if end is None:
        end = len(lines)

    results = []

    for i, line in enumerate(lines):
        if start <= i < end:
            verse_words = tokenize(line)
            overlap = verse_words & vocab
            results.append(",".join(sorted(overlap)))
            # print(f"Verse {i}: {line.strip()}")
            # print(f" → Overlap ({len(overlap)}): {sorted(overlap)}\n")
        else:
            results.append("")  # maintain 1:1 alignment with original lines

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out_file:
            for line in results:
                out_file.write(f"{line}\n")
        print(f"\nWrote overlap results to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare CSV vocabulary to each Bible verse (line) individually.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument("csv_column", help="Column name in the CSV file to extract vocabulary from")
    parser.add_argument("text_path", type=Path, help="Path to the plain text file (one verse per line)")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start line index in the text file (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End line index in the text file (exclusive)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to output .txt file with overlaps, one line per verse",
    )

    args = parser.parse_args()

    csv_vocab = get_csv_vocab(args.csv_path, args.csv_column)
    compare_and_write(args.text_path, csv_vocab, args.start, args.end, args.out)


if __name__ == "__main__":
    main()


# python compare_vocabularies.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASL_Citizen/splits/test.csv Gloss /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/corpus/eng-engULB.txt --start 0 --end 3 --out eng-engULB_vs_aslcitizen_vocab_overlap.txt

# ASL Citizen Vocab Overlap
# find /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/corpus/ -name "*eng-*.txt"|parallel -j4 python compare_vocabularies.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/ASL_Citizen/splits/test.csv Gloss "{}" --out "{/.}_vs_aslcitizen_vocab_overlap.txt"

# Sem-Lex Vocab Overlap
# find /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/corpus/ -name "*eng-*.txt"|parallel -j4 python compare_vocabularies.py /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/Sem-Lex/semlex_metadata.csv label "{}" --out ebible_isolatedsigndataset_vocab_overlaps/"{/.}_vs_semlex_vocab_overlap.txt"

# PopSign Vocab Overlap
# find /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/PopSignASL/test -mindepth 1 -maxdepth 1  -type d|cut -d "/" -f10 > /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/PopSignASL/foldernames.csv
# then manually add "NAME" at the top of the file...
#


# key terms list
# find /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/corpus/ -name "*eng-*.txt"|parallel -j4 python compare_vocabularies.py /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/unfolding_word_key_terms.csv TRANSLATION_WORD "{}" --out "{/.}_vs_unfoldingwordkeyterms_vocab_overlap.txt"
# conda activate /opt/home/cleong/envs/sign-bibles-dataset/ && cd /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign && find /opt/home/cleong/projects/semantic_and_visual_similarity/local_data/eBible/ebible/corpus/ -name "*eng-*.txt"|parallel -j4 python compare_vocabularies.py /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/unfolding_word_key_terms.csv TRANSLATION_WORD "{}" --out "{/.}_vs_unfoldingwordkeyterms_vocab_overlap.txt"
