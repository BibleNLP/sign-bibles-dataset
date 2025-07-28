import argparse
import io
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Video, get_dataset_config_names, load_dataset
from pose_format import Pose
from tqdm import tqdm


def iterate_over_dataset(language_subset: str, sample_count: int) -> pd.DataFrame:
    """Gather statistics about a dataset config and return a DataFrame of sample-level stats."""
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train")
    ds = ds.cast_column("mp4", Video())

    stats = []

    if sample_count is None:
        sample_iter = iter(ds)
    else:
        sample_iter = ds.take(sample_count)

    for sample in tqdm(sample_iter, desc=f"Parsing {language_subset}"):
        row = {
            "config": language_subset,
            "video_id": sample.get("__key__"),
        }

        # Load metadata + basic info
        metadata = sample.get("json", {})
        row["total_frames"] = metadata.get("total_frames", None)

        # Pose shape
        try:
            pose = Pose.read(io.BytesIO(sample["pose-mediapipe.pose"]))
            row["pose_frames"], row["pose_people"], row["pose_keypoints"], row["pose_xyz"] = pose.body.data.shape
        except Exception as e:
            row["pose_error"] = str(e)

        # Transcripts
        transcripts = sample.get("transcripts.json", [])
        row["num_transcripts"] = len(transcripts)
        row["num_transcripts_with_text"] = sum(1 for t in transcripts if t.get("text", "").strip())
        row["total_words"] = sum(len(t.get("text", "").strip().split()) for t in transcripts if t.get("text"))

        # Segment lengths
        segment_lengths = [
            t["end_frame"] - t["start_frame"]
            for t in transcripts
            if "start_frame" in t
            and "end_frame" in t
            and isinstance(t["start_frame"], int)
            and isinstance(t["end_frame"], int)
        ]
        row["avg_segment_length_frames"] = sum(segment_lengths) / len(segment_lengths) if segment_lengths else None

        stats.append(row)

    return pd.DataFrame(stats)


def main(sample_count: int, output_dir: Path, specific_subset: str | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_configs = [specific_subset] if specific_subset else get_dataset_config_names("bible-nlp/sign-bibles")
    all_dfs = []
    video_to_configs = defaultdict(set)

    for config in all_configs:
        df = iterate_over_dataset(language_subset=config, sample_count=sample_count)
        all_dfs.append(df)
        for vid in df["video_id"]:
            video_to_configs[vid].add(config)

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Add column for how many configs each video appears in
    full_df["video_configs_count"] = full_df["video_id"].map(lambda vid: len(video_to_configs[vid]))

    # Save to Parquet
    output_path = output_dir / "video_sample_stats.parquet"
    full_df.to_parquet(output_path, index=False)
    print(f"Saved stats to {output_path.resolve()}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Configs analyzed: {len(all_configs)}")
    print(f"Total unique videos: {len(set(full_df['video_id']))}")
    print(f"Videos reused in multiple configs: {(full_df['video_configs_count'] > 1).sum()}")
    print(f"Videos with transcripts: {(full_df['num_transcripts'] > 0).sum()}")
    print(f"Videos with at least one transcript with text: {(full_df['num_transcripts_with_text'] > 0).sum()}")
    print(f"Total transcripts: {full_df['num_transcripts'].sum()}")
    print(f"Total words: {full_df['total_words'].sum()}")
    valid_segments = full_df["avg_segment_length_frames"].dropna()
    if not valid_segments.empty:
        print(f"Average segment length (frames): {valid_segments.mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Sign Bibles dataset on Huggingface.")
    parser.add_argument("--language-subset", type=str, default=None, help="One config or leave blank to analyze all")
    parser.add_argument("--sample-count", type=int, help="Number of samples to analyze per config")
    parser.add_argument("--output-dir", type=Path, default="./stats_output/", help="Where to write stats parquet")

    args = parser.parse_args()
    main(sample_count=args.sample_count, output_dir=args.output_dir, specific_subset=args.language_subset)
