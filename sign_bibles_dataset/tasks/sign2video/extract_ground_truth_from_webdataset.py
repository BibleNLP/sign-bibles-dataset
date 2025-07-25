import argparse
from pathlib import Path

from datasets import get_dataset_config_names
from sign_language_gloss_utils.glosses.gloss_utils import get_dataset_vocab
from sign_language_gloss_utils.glosses.text_utils import get_glosses_set_from_text

from sign_bibles_dataset.tasks.text2video.extract_ground_truth_from_webdataset import extract_queries


def main(language_subset: str, sample_count: int, output_path: Path | None = None):
    queries_df = extract_queries(language_subset=language_subset, sample_count=sample_count)

    asl_citizen_vocab = get_dataset_vocab("asl-citizen")

    queries_df["query_glosses"] = queries_df["query_text"].apply(
        lambda text: get_glosses_set_from_text(text, asl_citizen_vocab)
    )

    queries_df = queries_df.drop(columns=["query_text"])  # query_text
    # Move 'query_glosses' to the front
    cols = queries_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("query_glosses")))
    queries_df = queries_df[cols]

    unique_video_ids = queries_df["video_id"].unique().tolist()

    default_csv_name = (
        Path(__file__).parent
        / "queries"
        / f"{language_subset}_{len(queries_df)}glossqueries_across_{len(unique_video_ids)}videos.csv"
    )
    if output_path is None:
        output_path = Path(default_csv_name)
    if output_path.is_dir():
        output_path = output_path / default_csv_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries_df["query_glosses"] = queries_df["query_glosses"].apply(lambda s: ",".join(sorted(s)))
    queries_df.to_json(output_path.with_suffix(".json"), orient="records", lines=True)
    queries_df.to_csv(output_path, index=False)

    print(f"Wrote queries to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Sign Bibles queries from Huggingface dataset.")
    parser.add_argument(
        "--language-subset",
        type=str,
        help="Language subset to download (default: all configs)",
    )
    parser.add_argument("--sample-count", type=int, default=10, help="Number of samples to parse (default: 10)")
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
