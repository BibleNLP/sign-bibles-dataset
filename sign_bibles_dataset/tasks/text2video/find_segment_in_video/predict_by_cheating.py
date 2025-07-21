import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


def generate_predictions_with_precision_recall(gt_df, row, recall, precision, num_predictions):
    total_frames = int(row["total_frames"])
    query_text, video_id = row["query_text"], row["video_id"]

    relevant_segments = gt_df[(gt_df["query_text"] == query_text) & (gt_df["video_id"] == video_id)][
        ["start_frame", "end_frame"]
    ].values.tolist()
    num_relevant_segments = len(relevant_segments)

    if num_relevant_segments == 0:
        raise ValueError(f"No relevant segments found for query '{query_text}' in video '{video_id}'.")

    unique_segments_to_hit = int(np.ceil(recall * num_relevant_segments))

    if unique_segments_to_hit > num_relevant_segments:
        raise ValueError(
            f"Impossible recall: need {unique_segments_to_hit} unique hits but only {num_relevant_segments} relevant segments."
        )

    min_required_hits = int(np.ceil(precision * num_predictions))
    if min_required_hits < unique_segments_to_hit:
        min_required_hits = unique_segments_to_hit

    if min_required_hits > num_predictions:
        raise ValueError(
            f"Impossible precision/recall: need at least {min_required_hits} hits to meet targets, but only {num_predictions} predictions allowed."
        )

    # Phase 1: unique segment hits to meet recall
    hit_segment_indices = list(
        np.random.choice(range(num_relevant_segments), size=unique_segments_to_hit, replace=False)
    )

    hits = []
    hit_segment_counter = Counter()

    for idx in hit_segment_indices:
        seg_start, seg_end = relevant_segments[idx]
        jitter = np.random.randint(-5, 5)
        start = max(0, seg_start + jitter)
        end = min(total_frames, seg_end + jitter)
        if start >= end:
            end = start + 1
        hits.append((start, end))
        hit_segment_counter[idx] += 1

    # Phase 2: repeat hits to meet precision
    additional_hits_needed = min_required_hits - len(hits)
    if additional_hits_needed > 0:
        additional_indices = np.random.choice(hit_segment_indices, size=additional_hits_needed, replace=True)
        for idx in additional_indices:
            seg_start, seg_end = relevant_segments[idx]
            jitter = np.random.randint(-5, 5)
            start = max(0, seg_start + jitter)
            end = min(total_frames, seg_end + jitter)
            if start >= end:
                end = start + 1
            hits.append((start, end))
            hit_segment_counter[idx] += 1

    # Phase 3: misses
    num_misses = num_predictions - len(hits)
    misses = []
    for _ in range(num_misses):
        while True:
            start, end = np.sort(np.random.randint(0, total_frames, size=2))
            if end - start < 2:
                end = start + 2
            if all(end <= seg_start or start >= seg_end for seg_start, seg_end in relevant_segments):
                misses.append((start, end))
                break

    predictions = hits + misses
    np.random.shuffle(predictions)

    formatted_preds = []
    for rank, (start, end) in enumerate(predictions, 1):
        formatted_preds.append(
            {
                "query_text": query_text,
                "video_id": video_id,
                "start_frame": int(start),
                "end_frame": int(end),
                "rank": rank,
            }
        )

    hit_segments_used = set(hit_segment_indices)
    print(f"Query '{query_text}' (video_id={video_id}):")
    print(f"  Ground Truth Segments ({num_relevant_segments}): {[f'{s}-{e}' for s, e in relevant_segments]}")
    print(f"  Requested Recall: {recall:.2f}, Precision: {precision:.2f}, Predictions: {num_predictions}")
    proportion_of_unique_segments_hit = len(hit_segments_used) / num_relevant_segments
    print(
        f"  Unique Segments Hit: {len(hit_segments_used)}/{num_relevant_segments}, or Recall = {proportion_of_unique_segments_hit:.2f}"
    )
    hits_proportion = len(hits) / num_predictions
    print(f"  Hits Generated: {len(hits)}, Misses Generated: {len(misses)}, or Precision = {hits_proportion}")
    print("  Hit Segment Usage:")
    for seg_idx, count in hit_segment_counter.items():
        seg_start, seg_end = relevant_segments[seg_idx]
        print(f"    Segment {seg_start}-{seg_end}: {count} times")

    return formatted_preds


def main(gt_path: Path, out_path: Path, recall: float, precision: float, num_predictions: int):
    gt_df = pd.read_csv(gt_path)
    all_preds = []

    unique_queries = gt_df.drop_duplicates(subset=["query_text", "video_id"])

    for _, row in unique_queries.iterrows():
        preds = generate_predictions_with_precision_recall(gt_df, row, recall, precision, num_predictions)
        all_preds.extend(preds)

    out_df = pd.DataFrame(all_preds)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} predictions with recall={recall} and precision={precision} to {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions with controlled precision and recall from ground truth."
    )
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground truth CSV.")
    parser.add_argument("--output-path", type=Path, default=Path("cheating_predictions.csv"))
    parser.add_argument("--recall", type=float, default=1.0, help="Recall (0-1).")
    parser.add_argument("--precision", type=float, default=1.0, help="Precision (0-1).")
    parser.add_argument("--num-predictions", type=int, default=10, help="Number of predictions per query.")
    args = parser.parse_args()

    main(
        gt_path=args.ground_truth,
        out_path=args.output_path,
        recall=args.recall,
        precision=args.precision,
        num_predictions=args.num_predictions,
    )
