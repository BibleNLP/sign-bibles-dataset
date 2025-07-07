#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()


def get_possible_asl_lex_matches(bible_text_list, asl_lex_vocab):
    bible_text_to_check = " ".join(bible_text_list)

    # strip alphanumeric
    bible_text_to_check = re.sub(r"[^a-zA-Z\s]", "", bible_text_to_check)

    asl_lex_possible_matches = set()

    for word in bible_text_to_check.split():
        # lowercase the word
        word = word.lower()
        for asl_lex_word in asl_lex_vocab:
            # check for exact match or word followed by underscore + number
            if asl_lex_word == word or re.fullmatch(
                rf"{re.escape(word)}_\d+", asl_lex_word
            ):
                # print(word, asl_lex_word)
                asl_lex_possible_matches.add(asl_lex_word)
    return asl_lex_possible_matches


def main():
    parser = argparse.ArgumentParser(description="Read JSON and CSV files into memory.")
    parser.add_argument(
        "json_path", help="Path to the JSON file with updated Bible verse info"
    )
    parser.add_argument("csv_path", help="Path to the asllex_signdata.csv file")
    args = parser.parse_args()

    # Load the JSON data
    with open(args.json_path, encoding="utf-8") as f:
        json_data = json.load(f)
    print(f"Loaded JSON with {len(json_data)} top-level groups, here's the head:")
    signbible_df = pd.json_normalize(
        json_data,
        record_path="videos",
        meta=["language_code", "version_name", "dbl_id"],
    )
    ase_df = signbible_df[signbible_df["language_code"] == "ase"]

    print(
        "Here are the portions which have language code ase, aka American Sign Language"
    )
    print(ase_df.head())
    print(ase_df.info())
    # ase_df = ase_df.head()

    # parse mp4_path filename to "video_title" column, parent folder to "Project name"
    # parse mp4_path filename to "video_title" column, parent folder to "Project name"
    if "mp4_path" in ase_df.columns:
        ase_df["video_title"] = ase_df["mp4_path"].apply(lambda p: Path(p).name)
        ase_df["project_name"] = ase_df["mp4_path"].apply(lambda p: Path(p).parent.name)
        print("Extracted 'video_title' and 'Project name' from 'mp4_path'")
        print(ase_df[["project_name", "video_title"]].head())
    else:
        print("Warning: 'mp4_path' column not found in the data")

    # Load the CSV data
    asllex_df = pd.read_csv(args.csv_path, encoding="latin-1")
    asl_lex_vocab = asllex_df["EntryID"].unique().tolist()
    print(f"Loaded CSV with {len(asllex_df)} rows and {len(asllex_df.columns)} columns")

    print(f"Vocab from ASL LEX: {len(asl_lex_vocab)} glosses")

    print(ase_df["bible_text"])

    ase_df["asllex_gloss_candidates"] = ase_df["bible_text"].progress_apply(
        lambda x: get_possible_asl_lex_matches(x, asl_lex_vocab)
        if isinstance(x, list)
        else set()
    )
    print(ase_df["asllex_gloss_candidates"])
    output_path = "ase_augmented_with_gloss_candidates.json"
    ase_df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved augmented DataFrame to {output_path}")


if __name__ == "__main__":
    main()
