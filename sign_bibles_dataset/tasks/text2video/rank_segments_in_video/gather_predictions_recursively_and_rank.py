#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from sign_bibles_dataset.tasks.text2video.rank_segments_in_video.evaluate_predictions import evaluate_prediction


def load_and_concat_csvs(files: list[Path], drop_duplicates: bool = False) -> pd.DataFrame:
    dfs = [pd.read_csv(f) for f in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    if drop_duplicates:
        combined_df = combined_df.drop_duplicates()
    return combined_df


def main(input_dir: Path, output_csv: Path):
    if not input_dir.is_dir():
        raise ValueError(f"Provided path is not a directory: {input_dir}")

    if output_csv is None:
        output_csv = input_dir / "evaluation_results.csv"

    # Find all matching files
    ground_truth_files = list(input_dir.rglob("*ground_truth.csv"))
    prediction_files = list(input_dir.rglob("*all_predictions.csv"))

    print(f"Found {len(ground_truth_files)} ground truth files.")
    print(f"Found {len(prediction_files)} prediction files.")

    # Combine ground truth (dropping duplicates)
    combined_gt = load_and_concat_csvs(ground_truth_files, drop_duplicates=True)
    combined_gt_out = Path("combined_ground_truth.csv")
    combined_gt.to_csv(combined_gt_out, index=False)
    print(f"Wrote {combined_gt_out.resolve()}")

    # Combine predictions (no deduplication)
    combined_preds = load_and_concat_csvs(prediction_files, drop_duplicates=False)
    combined_preds_out = Path("combined_predictions_for_all_videos.csv")
    combined_preds.to_csv(combined_preds_out, index=False)
    print(f"Wrote \n{combined_preds_out.resolve()}")

    results_df = evaluate_prediction(combined_gt_out, combined_preds_out)
    print(results_df)
    print(f"Saved evaluation results to {Path(output_csv).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine ground truth and prediction CSVs.")
    parser.add_argument("input_dir", type=Path, help="Path to directory containing CSVs")
    parser.add_argument(
        "--output_csv",
        type=Path,
        # default="evaluation_results.csv",
        help="Where to save results (default: same as the input dir)",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)


