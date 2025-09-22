import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Video, get_dataset_config_names, load_dataset
from pose_format import Pose
from tqdm import tqdm


def calculate_autosegment_stats(autosegmented_segments_dict):
    collector = defaultdict(list)

    for tier_name, segments in autosegmented_segments_dict.items():
        for segment in segments:
            start_ms = segment["start_ms"]
            end_ms = segment["end_ms"]

            duration_ms = end_ms - start_ms
            assert duration_ms >= 0

            collector["tier_name"].append(tier_name)
            collector["duration_ms"].append(duration_ms)

    segments_df = pd.DataFrame(collector)
    print(segments_df.head())

    # print(seg)

    # Compute average and count of duration_ms per tier
    grouped = segments_df.groupby("tier_name")["duration_ms"]
    stats_dict = {tier: {"avg": group.mean(), "count": group.count()} for tier, group in grouped}

    return stats_dict


def iterate_over_dataset(language_subset: str, split: str, sample_count: int | None) -> pd.DataFrame:
    """Gather statistics about a dataset config and split; return a DataFrame of sample-level stats."""
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split=split)
    ds = ds.cast_column("mp4", Video())
    stats = []

    sample_iter = iter(ds) if sample_count is None else ds.take(sample_count)

    for sample in tqdm(sample_iter, desc=f"Parsing {language_subset}:{split}"):
        row = {
            "config": language_subset,
            "split": split,
            "video_id": sample.get("__key__"),
        }

        metadata = sample.get("json", {})
        row["total_frames"] = metadata.get("total_frames", None)

        try:
            pose = Pose.read(io.BytesIO(sample["pose-mediapipe.pose"]))
            row["pose_frames"], row["pose_people"], row["pose_keypoints"], row["pose_xyz"] = pose.body.data.shape
        except Exception as e:
            row["pose_error"] = str(e)

        transcripts = sample.get("transcripts.json", [])
        row["num_transcripts"] = len(transcripts)
        row["num_transcripts_with_text"] = sum(1 for t in transcripts if t.get("text", "").strip())
        row["total_words"] = sum(len(t.get("text", "").strip().split()) for t in transcripts if t.get("text"))

        segment_lengths = [
            t["end_frame"] - t["start_frame"]
            for t in transcripts
            if isinstance(t.get("start_frame"), int) and isinstance(t.get("end_frame"), int)
        ]
        row["avg_segment_length_frames"] = sum(segment_lengths) / len(segment_lengths) if segment_lengths else None

        keys = sample.keys()
        print(keys)

        autosegmented_segments_dict = sample["model_e4s-1.autosegmented_segments.json"]
        print(json.dumps(autosegmented_segments_dict, indent=2))
        autosegment_stats = calculate_autosegment_stats(autosegmented_segments_dict)
        # print(json.dumps(autosegment_stats, indent=2))
        # autosegment_stats.keys()
        for key, value in autosegment_stats.items():
            row[key] = value

        stats.append(row)

    return pd.DataFrame(stats)


def main(sample_count: int | None, output_dir: Path, specific_subset: str | None = None, split: str | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_configs = [specific_subset] if specific_subset else get_dataset_config_names("bible-nlp/sign-bibles")
    all_splits = [split] if split else ["train", "val", "test"]
    all_dfs = []
    video_to_configs = defaultdict(set)

    for config in all_configs:
        for split_name in all_splits:
            # try:
            df = iterate_over_dataset(language_subset=config, split=split_name, sample_count=sample_count)
            # except Exception as e:
            #     print(f"⚠️ Skipping {config}:{split_name} — {type(e)}: {e}")
            #     continue

            all_dfs.append(df)
            for vid in df["video_id"]:
                video_to_configs[vid].add(f"{config}:{split_name}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df["video_configsplits_count"] = full_df["video_id"].map(lambda vid: len(video_to_configs[vid]))

    output_path = output_dir / "video_sample_stats.parquet"
    full_df.to_parquet(output_path, index=False)
    print(f"Saved stats to {output_path.resolve()}")

    print("\n=== SUMMARY ===")
    print(f"Configs analyzed: {len(all_configs)}")
    print(f"Splits analyzed: {all_splits}")
    print(f"Total unique videos: {len(set(full_df['video_id']))}")
    print(f"Videos reused in multiple config/splits: {(full_df['video_configsplits_count'] > 1).sum()}")
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
    parser.add_argument("--sample-count", type=int, help="Number of samples to analyze per config/split")
    parser.add_argument(
        "--split", type=str, default=None, help="One split (train/validation/test) or leave blank for all"
    )
    parser.add_argument("--output-dir", type=Path, default="./stats_output/", help="Where to write stats parquet")
    args = parser.parse_args()

    main(
        sample_count=args.sample_count,
        output_dir=args.output_dir,
        specific_subset=args.language_subset,
        split=args.split,
    )
