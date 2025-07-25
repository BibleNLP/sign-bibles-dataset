import argparse
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

from sign_bibles_dataset.dataprep.dbl_signbibles.ebible_utils.vref_lookup import (
    get_bible_verse_by_vref_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_queries(language_subset: str, sample_count: int | None) -> pd.DataFrame:
    """Extract valid queries from the Sign Bibles dataset."""
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train").decode(False)
    logging.info(f"Loaded dataset subset '{language_subset}'")

    queries = defaultdict(list)

    iterable = itertools.islice(ds, sample_count)
    desc = f"Parsing {'all' if sample_count is None else sample_count} samples"

    for sample_idx, sample in enumerate(tqdm(iterable, desc=desc)):
        # print(f"Parsing sample {sample_idx}: {sample['__key__']}")
        if "transcripts.json" in sample:
            transcripts = sample["transcripts.json"]
            if len(transcripts) <= 1:
                continue

            total_frames = sample["json"]["total_frames"]
            segments = defaultdict(list)

            for transcript in transcripts:
                start_frame = transcript["start_frame"]
                end_frame = transcript["end_frame"]
                for vref_index in transcript["biblenlp-vref"]:
                    verse_text = get_bible_verse_by_vref_index(vref_index)
                    segments[(start_frame, end_frame)].append(verse_text)

            sorted_segments = sorted(segments.items(), key=lambda x: x[0][0])

            for seg_idx, ((start_frame, end_frame), verse_texts) in enumerate(sorted_segments):
                for verse_text in verse_texts:
                    if verse_text:
                        queries["query_text"].append(verse_text)
                        queries["video_id"].append(sample["__key__"])
                        queries["seg_idx"].append(seg_idx)
                        queries["start_frame"].append(start_frame)
                        queries["end_frame"].append(end_frame)
                        queries["total_frames"].append(total_frames)

    return pd.DataFrame(queries)


def main(language_subset: str, sample_count: int, output_path: Path | None = None):
    queries_df = extract_queries(language_subset=language_subset, sample_count=sample_count)
    unique_video_ids = queries_df["video_id"].unique().tolist()

    default_csv_name = (
        Path(__file__).parent
        / "queries"
        / f"{language_subset}_{len(queries_df)}queries_across_{len(unique_video_ids)}videos.csv"
    )
    if output_path is None:
        output_path = Path(default_csv_name)
    if output_path.is_dir():
        output_path = output_path / default_csv_name
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
    parser.add_argument("--sample-count", default=10, type=int, help="Number of samples to parse (default: 10)")
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Path to output CSV. Default: {language_subset}_{sample_count}_queries.csv",
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

        main(
            language_subset=language_subset,
            sample_count=args.sample_count,
            output_path=args.output_path,
        )
