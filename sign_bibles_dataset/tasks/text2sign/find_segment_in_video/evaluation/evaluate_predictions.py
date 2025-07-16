import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_iou(gt_start, gt_end, pred_start, pred_end) -> float:
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return intersection / union if union > 0 else 0.0


def compute_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame, recall_ks: list[int]) -> dict:
    results = defaultdict(float)
    queries = gt_df[["query_text", "video_id"]].drop_duplicates()
    all_ious = []
    recall_counts = {k: 0 for k in recall_ks}

    for _, query in queries.iterrows():
        q_text, v_id = query["query_text"], query["video_id"]
        gt = gt_df[(gt_df["query_text"] == q_text) & (gt_df["video_id"] == v_id)]
        preds = pred_df[(pred_df["query_text"] == q_text) & (pred_df["video_id"] == v_id)]

        if gt.empty or preds.empty:
            continue

        gt_start, gt_end = gt.iloc[0]["start_frame"], gt.iloc[0]["end_frame"]
        ious = [
            calculate_iou(gt_start, gt_end, p_start, p_end)
            for p_start, p_end in preds[["start_frame", "end_frame"]].values
        ]

        all_ious.append(max(ious))

        for k in recall_ks:
            topk_ious = sorted(ious, reverse=True)[:k]
            if any(iou > 0 for iou in topk_ious):
                recall_counts[k] += 1

    total = len(all_ious)
    results["miou"] = np.mean(all_ious) if total else 0.0
    for k in recall_ks:
        results[f"recall@{k}"] = recall_counts[k] / total if total else 0.0

    print(f"Evaluated {total} queries.")
    return results


def save_metrics(metrics: dict, predictor_name: str, metrics_log_path: Path):
    metrics_row = {"predictor": predictor_name, **metrics}
    metrics_df = pd.DataFrame([metrics_row])

    if metrics_log_path.exists():
        existing_df = pd.read_csv(metrics_log_path)
        combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        combined_df = metrics_df

    combined_df.to_csv(metrics_log_path, index=False)
    print(f"Appended metrics to {metrics_log_path.resolve()}")


def main(gt_path: Path, pred_path: Path, recall_ks: list[int], predictor_name: str, metrics_log_path: Path):
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    metrics = compute_metrics(gt_df, pred_df, recall_ks)

    print("\nMetrics:")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

    save_metrics(metrics, predictor_name, metrics_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions with mIoU and Recall@K.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground truth CSV.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions CSV.")
    parser.add_argument("--predictor-name", type=str, required=True, help="Name of predictor/model to log.")
    parser.add_argument(
        "--metrics-log-path",
        type=Path,
        default=Path("metrics_log.csv"),
        help="CSV file to append metrics (default: metrics_log.csv)",
    )
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[1, 5], help="Recall@K values (default: 1 5)")

    args = parser.parse_args()

    main(
        gt_path=args.ground_truth,
        pred_path=args.predictions,
        recall_ks=args.recall_ks,
        predictor_name=args.predictor_name,
        metrics_log_path=args.metrics_log_path,
    )
