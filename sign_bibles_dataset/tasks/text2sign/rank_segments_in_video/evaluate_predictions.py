import argparse
import pandas as pd
from pathlib import Path

import torch
from torchmetrics.retrieval import (
    RetrievalRecall,
    RetrievalPrecision,
    RetrievalMRR,
    RetrievalMAP,
    RetrievalNormalizedDCG,
)


def prepare_references(df: pd.DataFrame) -> dict[str, set[int]]:
    """Group ground truth seg_idx by query_text"""
    return df.groupby("query_text")["seg_idx"].apply(set).to_dict()


def load_predictions(pred_path: Path, k: int) -> dict[str, list[int]]:
    """Load flat prediction CSV: one row per prediction"""
    df = pd.read_csv(pred_path)
    df = df.sort_values(["query_text", "rank"])
    grouped = df.groupby("query_text")["seg_idx"].apply(list)
    return {query: segs[:k] for query, segs in grouped.items()}


def align_predictions_and_refs(preds: dict, refs: dict, k: int):
    """Return flat tensors: scores, binary targets, and query indexes"""
    scores = []
    targets = []
    query_ids = []

    idx_counter = 0
    for query in sorted(set(preds) & set(refs)):
        pred_list = preds[query][:k]
        ref_set = refs[query]

        for seg_idx in pred_list:
            targets.append(int(seg_idx in ref_set))
            scores.append(1.0)  # uniform score for ranked order
            query_ids.append(idx_counter)
        idx_counter += 1

    return (
        torch.tensor(scores, dtype=torch.float),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(query_ids, dtype=torch.long),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranked predictions using TorchMetrics")
    parser.add_argument("ground_truth_csv", type=Path, help="CSV with seg_idx and query_text")
    parser.add_argument("predictions_csv", type=Path, help="CSV with query_text,rank,seg_idx")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--output_csv", type=Path, default="evaluation_results.csv", help="Where to save results")

    args = parser.parse_args()

    gt_df = pd.read_csv(args.ground_truth_csv)
    refs = prepare_references(gt_df)
    max_k = max(args.ks)
    print(f"K values: {args.ks}, Max K: {max_k}")
    preds = load_predictions(args.predictions_csv, max_k)

    scores, targets, query_ids = align_predictions_and_refs(preds, refs, max_k)

    # Initialize metrics
    metrics = {}
    for k in args.ks:
        metrics[f"precision_at_{k}"] = RetrievalPrecision(top_k=k)
        metrics[f"recall_at_{k}"] = RetrievalRecall(top_k=k)

    metrics["MRR"] = RetrievalMRR()
    metrics["MAP"] = RetrievalMAP()
    metrics["NDCG"] = RetrievalNormalizedDCG()

    # Evaluate
    metric_results = {}
    for metric_name, metric in metrics.items():
        # returns a single-element tensor
        metric_results[metric_name] = metric(scores, targets, indexes=query_ids).item()
    # Save results
    results_df = pd.DataFrame([metric_results])

    results_df.to_csv(args.output_csv, index=False)
    print(results_df)
    print(f"Saved evaluation results to {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
