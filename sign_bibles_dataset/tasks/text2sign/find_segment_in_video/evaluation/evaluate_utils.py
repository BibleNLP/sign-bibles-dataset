import numpy as np
import pandas as pd

Segment = tuple[int, int]  # (start_frame, end_frame)


def calculate_iou(seg1: Segment, seg2: Segment) -> float:
    start1, end1 = seg1
    start2, end2 = seg2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0.0


def get_retrieved_segments(
    predictions: list[Segment], ground_truth: list[Segment], iou_threshold: float = 0.1
) -> set[int]:
    """Return indices of GT segments hit by any prediction."""
    retrieved = set()
    for idx, gt in enumerate(ground_truth):
        for pred in predictions:
            if calculate_iou(pred, gt) >= iou_threshold:
                retrieved.add(idx)
                break
    return retrieved


def get_relevant_segments(ground_truth_df: pd.DataFrame, query_text: str, video_id: str) -> list[tuple[int, int]]:
    """Return the list of (start_frame, end_frame) for the given query."""
    gt_rows = ground_truth_df[(ground_truth_df["query_text"] == query_text) & (ground_truth_df["video_id"] == video_id)]
    return list(gt_rows[["start_frame", "end_frame"]].itertuples(index=False, name=None))


def count_relevant_segments(ground_truth_df: pd.DataFrame, query_text: str, video_id: str) -> int:
    """Count how many ground truth segments exist for a query."""
    return len(get_relevant_segments(ground_truth_df, query_text, video_id))


def calculate_precision(predictions: list[Segment], ground_truth: list[Segment], iou_threshold: float = 0.1) -> float:
    if not predictions:
        return 0.0
    hits = sum(any(calculate_iou(pred, gt) >= iou_threshold for gt in ground_truth) for pred in predictions)
    return hits / len(predictions)


def calculate_recall(predictions: list[Segment], ground_truth: list[Segment], iou_threshold: float = 0.1) -> float:
    if not ground_truth:
        return 0.0
    retrieved = get_retrieved_segments(predictions, ground_truth, iou_threshold)
    return len(retrieved) / len(ground_truth)


def calculate_miou(predictions: list[Segment], ground_truth: list[Segment]) -> float:
    if not predictions or not ground_truth:
        return 0.0
    ious = [max(calculate_iou(pred, gt) for gt in ground_truth) for pred in predictions]
    return np.mean(ious)


def calculate_redundancy(predictions: list[Segment], iou_threshold: float = 0.1) -> float:
    """Measures prediction redundancy: proportion of predictions overlapping each other."""
    if len(predictions) < 2:
        return 0.0
    overlaps = 0
    total_pairs = len(predictions) * (len(predictions) - 1) / 2
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            if calculate_iou(predictions[i], predictions[j]) >= iou_threshold:
                overlaps += 1
    return overlaps / total_pairs
