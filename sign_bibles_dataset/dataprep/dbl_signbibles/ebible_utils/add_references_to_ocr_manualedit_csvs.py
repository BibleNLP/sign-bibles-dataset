import argparse
from pathlib import Path

import pandas as pd
from vref_lookup import (
    citation_to_text_and_vrefs,
    load_bible_lines,
    load_usfm_book_map,
    load_vref_map,
)


def main():
    parser = argparse.ArgumentParser(description="Apply citation parser to ocr.manualedit.csv files.")
    parser.add_argument("input_dir", type=Path, help="Directory to search recursively for CSV files")
    args = parser.parse_args()

    input_dir: Path = args.input_dir

    # Load necessary resources
    base_dir = Path(__file__).parent / "data"
    vref_path = base_dir / "vref.txt"
    bible_path = base_dir / "eng-engbsb.txt"
    usfm_csv_path = base_dir / "usfm_book_identifiers.csv"

    vref_map = load_vref_map(vref_path)
    bible_lines = load_bible_lines(bible_path)
    book_map = load_usfm_book_map(usfm_csv_path)

    # Search for target files
    for csv_path in input_dir.rglob("*ocr.manualedit.csv"):
        print(f"Processing: {csv_path}")
        df = pd.read_csv(csv_path)

        # Apply function to 'text' column
        def extract(row):
            try:
                bible_text, vrefs = citation_to_text_and_vrefs(
                    citation=row["text"],
                    vref_map=vref_map,
                    bible_lines=bible_lines,
                    book_map=book_map,
                )
                return pd.Series({"bible_text": bible_text, "vrefs": vrefs})
            except Exception:
                return pd.Series({"bible_text": None, "vrefs": None})

        result_df = df.join(df.apply(extract, axis=1))
        # print(result_df[["text", "bible_text", "vrefs"]].head())  # Preview result
        # print(result_df[["text", "bible_text"]].head())  # Preview result

        # Optionally, save result
        output_path = csv_path.with_suffix(".withvrefs.csv")
        print(f"Saving to {output_path}")
        result_df.to_csv(csv_path.with_suffix(".withvrefs.csv"), index=False)


if __name__ == "__main__":
    main()
