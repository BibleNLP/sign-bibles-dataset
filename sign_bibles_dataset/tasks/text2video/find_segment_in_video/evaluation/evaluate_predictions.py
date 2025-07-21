import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_iou(gt_start, gt_end, pred_start, pred_end):
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return intersection / union if union > 0 else 0.0


def compute_metrics(gt_df, pred_df, recall_ks, precision_ks):
    results = defaultdict(float)
    queries = gt_df[["query_text", "video_id"]].drop_duplicates()
    all_best_ious = []
    recall_at_k_counts = {k: 0 for k in recall_ks}
    precision_at_k_values = {k: [] for k in precision_ks}

    print(f"\n---- EVALUATION START ----\n")

    for _, row in queries.iterrows():
        q_text, v_id = row["query_text"], row["video_id"]

        gt_segments = gt_df.query("query_text == @q_text and video_id == @v_id")[["start_frame", "end_frame"]].values
        preds = pred_df.query("query_text == @q_text and video_id == @v_id")[["start_frame", "end_frame"]].values

        if len(preds) == 0:
            print(f"⚠️ No predictions for ({q_text}, {v_id}), counting as zero recall and precision.")
            all_best_ious.append(0.0)
            for k in recall_ks:
                recall_at_k_counts[k] += 0
            for k in precision_ks:
                precision_at_k_values[k].append(0.0)
            continue

        # Compute IoUs for each prediction
        pred_ious = []
        for p_start, p_end in preds:
            ious = [calculate_iou(gt_start, gt_end, p_start, p_end) for gt_start, gt_end in gt_segments]
            pred_ious.append(max(ious))

        # Mean IoU for best prediction
        best_iou = max(pred_ious)
        all_best_ious.append(best_iou)

        print(f"Query '{q_text}' ({v_id}): {len(gt_segments)} GT segments, {len(preds)} predictions")
        print(f"  Best IoU = {best_iou:.4f}")

        sorted_ious = sorted(pred_ious, reverse=True)

        # Recall@K: does at least 1 of top-K predictions overlap with GT?
        for k in recall_ks:
            topk_hits = sorted_ious[:k]
            if any(iou > 0 for iou in topk_hits):
                recall_at_k_counts[k] += 1
                print(f"  ✅ Recall@{k}: HIT")
            else:
                print(f"  ❌ Recall@{k}: MISS")

        # Precision@K: proportion of relevant (IoU > 0) in top-K
        for k in precision_ks:
            topk_hits = sorted_ious[:k]
            precision = sum(iou > 0 for iou in topk_hits) / len(topk_hits)
            precision_at_k_values[k].append(precision)
            print(f"  Precision@{k}: {precision:.3f}")

    total_queries = len(queries)
    results["miou"] = np.mean(all_best_ious) if total_queries else 0.0
    for k in recall_ks:
        results[f"recall@{k}"] = recall_at_k_counts[k] / total_queries
    for k in precision_ks:
        results[f"precision@{k}"] = np.mean(precision_at_k_values[k])

    print(f"\nEvaluated {total_queries} queries.")
    print(f"Mean IoU = {results['miou']:.4f}")
    for k in recall_ks:
        print(f"Recall@{k} = {results[f'recall@{k}']:.4f}")
    for k in precision_ks:
        print(f"Precision@{k} = {results[f'precision@{k}']:.4f}")

    print("\n---- EVALUATION END ----\n")
    return results


def save_metrics(metrics, predictor_name, metrics_log_path):
    metrics_row = {"predictor": predictor_name, **metrics}
    metrics_df = pd.DataFrame([metrics_row])

    if metrics_log_path.exists():
        existing_df = pd.read_csv(metrics_log_path)
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        combined_df = metrics_df

    combined_df.to_csv(metrics_log_path, index=False)
    print(f"Appended metrics to {metrics_log_path.resolve()}")


def main(gt_path, pred_path, recall_ks, precision_ks, predictor_name, metrics_log_path):
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    print(f"Read {len(gt_df)} ground truth queries from {gt_path}")
    print(f"Read {len(pred_df)} predictions from {pred_path}")

    metrics = compute_metrics(gt_df, pred_df, recall_ks, precision_ks)

    print("\nFinal Metrics:")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

    save_metrics(metrics, predictor_name, metrics_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval predictions with Recall@K and Precision@K metrics."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictor-name", type=str, required=True)
    parser.add_argument("--metrics-log-path", type=Path, default=Path("metrics_log.csv"))
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--precision-ks", type=int, nargs="+", default=[1, 5, 10])

    args = parser.parse_args()

    main(
        gt_path=args.ground_truth,
        pred_path=args.predictions,
        recall_ks=args.recall_ks,
        precision_ks=args.precision_ks,
        predictor_name=args.predictor_name,
        metrics_log_path=args.metrics_log_path,
    )


# python sign_bibles_dataset/tasks/text2sign/find_segment_in_video/evaluation/evaluate_predictions.py --ground-truth sign_bibles_dataset/tasks/text2sign/find_segment_in_video/examples/example_queries.csv --predictions "sign_bibles_dataset/tasks/text2sign/find_segment_in_video/examples/example_random_predictions.csv" --predictor-name "random_guessing" --metrics-log-path sign_bibles_dataset/tasks/text2sign/find_segment_in_video/examples/metrics_log.csv


# python sign_bibles_dataset/tasks/text2sign/find_segment_in_video/evaluation/evaluate_predictions.py --ground-truth sign_bibles_dataset/tasks/text2sign/find_segment_in_video/examples/example_queries.csv --predictions "sign_bibles_dataset/tasks/text2sign/find_segment_in_video/examples/wrong_predictions.csv" --predictor-name "always_wrong" --metrics-log-path sign_bibles_dataset/tasks/text2sign/find_segment_in_video/examples/metrics_log.csv
