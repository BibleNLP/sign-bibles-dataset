import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

from sign_bibles_dataset.dataprep.dbl_signbibles.ebible_utils.vref_lookup import get_bible_verse_by_vref_index


def extract_queries(language_subset: str, sample_count: int) -> pd.DataFrame:
    """Extract valid queries from the Sign Bibles dataset."""
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train")
    ds = ds.decode(False)  # skip decoding video files
    print(f"Loaded dataset subset '{language_subset}'")

    queries = defaultdict(list)

    for sample in tqdm(ds.take(sample_count), desc="Parsing samples"):
        transcripts = sample["json"].get("transcripts", [])
        if transcripts:
            total_frames = sample["json"]["total_frames"]
            for transcript in transcripts:
                start_frame = transcript["start_frame"]
                end_frame = transcript["end_frame"]
                for seg_idx, vref_index in enumerate(transcript["biblenlp-vref"]):
                    verse_text = get_bible_verse_by_vref_index(vref_index)
                    queries["seg_idx"].append(seg_idx)
                    queries["query_text"].append(verse_text)
                    queries["video_id"].append(sample["__key__"])
                    queries["start_frame"].append(start_frame)
                    queries["end_frame"].append(end_frame)
                    queries["total_frames"].append(total_frames)

    queries_df = pd.DataFrame(queries)
    print(f"Extracted {len(queries_df)} queries.")
    return queries_df


def main(language_subset: str, sample_count: int, output_path: Path):
    queries_df = extract_queries(language_subset=language_subset, sample_count=sample_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queries_df.to_csv(output_path, index=False)
    print(f"Wrote queries to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Sign Bibles queries from Huggingface dataset.")
    parser.add_argument(
        "--language-subset",
        type=str,
        help="Language subset to download (default: all configs)",
    )
    parser.add_argument("--sample-count", type=int, default=20, help="Number of samples to parse (default: 5)")
    parser.add_argument(
        "--output-path", type=Path, help="Path to output CSV. Default: {language_subset}_{sample_count}_queries.csv"
    )

    args = parser.parse_args()

    if args.language_subset is not None:
        configs = [args.language_subset]
    else:
        configs = get_dataset_config_names("bible-nlp/sign-bibles")
        print(f"Available configs: {configs}")

    for language_subset in configs:
        print(f"Extracting from subset {language_subset}")
        # Set default output path if not provided
        output_path = args.output_path or Path(f"{language_subset}_{args.sample_count}_queries.csv")

        main(language_subset=language_subset, sample_count=args.sample_count, output_path=output_path)
