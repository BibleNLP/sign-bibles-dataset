#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from sign_bibles_dataset.tasks.text2video.rank_segments_in_video.evaluate_predictions import evaluate_prediction
from sign_bibles_dataset.tasks.text2video.rank_segments_in_video.predict_by_cheating import generate_predictions


def load_and_concat_csvs(files: list[Path], drop_duplicates: bool = False) -> pd.DataFrame:
    dfs = [pd.read_csv(f) for f in files]
    combined_df = pd.concat(dfs, ignore_index=True)
    if drop_duplicates:
        combined_df = combined_df.drop_duplicates()
    return combined_df


def main(input_dir: Path, output_csv: Path, generate_random: bool = True):
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
    combined_gt_out = input_dir / "combined_ground_truth.csv"
    combined_gt.to_csv(combined_gt_out, index=False)
    print(f"Wrote {combined_gt_out.resolve()}")

    # Combine predictions (no deduplication)
    combined_preds = load_and_concat_csvs(prediction_files, drop_duplicates=False)
    combined_preds_out = input_dir / "combined_predictions_for_all_videos.csv"
    combined_preds.to_csv(combined_preds_out, index=False)
    print(f"Wrote \n{combined_preds_out.resolve()}")

    results_df = evaluate_prediction(combined_gt_out, combined_preds_out)

    print(f"Saved evaluation results to {Path(output_csv).resolve()}")

    if generate_random:
        random_results_collector = []
        for i in range(30):
            k = int(results_df["avg_segments_ranked"][0])
            predictions_df = generate_predictions(combined_gt, k, mode="random")
            predictions_df_out = input_dir / f"random_basedline_predictions_{i}.csv"
            predictions_df.to_csv(predictions_df_out)

            random_results_eval_df = evaluate_prediction(combined_gt_out, predictions_df_out)
            random_results_collector.append(random_results_eval_df)
        print("RANDOM PREDICTIONS:")
        random_eval = pd.concat(random_results_collector)
        print(random_eval)
        random_eval = random_eval.drop(columns=["precision_at_10", "recall_at_10"])

        print(f"AVERAGE OF RANDOM")
        random_eval_mean_df = random_eval.mean(numeric_only=True).to_frame().T
        print(random_eval_mean_df)

        # print(random_eval.describe())

    results_df = results_df.rename(
        columns={
            "precision_at_1": "P@1",
            "recall_at_1": "R@1",
            "precision_at_5": "P@5",
            "recall_at_5": "R@5",
            "avg_segments_ranked": "Avg Segs",
            # "precision_at_10": "P@10",
            # "recall_at_10": "R@10",
        }
    )

    results_df = results_df.drop(columns=["precision_at_10", "recall_at_10"])
    results_df["Window"] = input_dir.name.split("_")[0].split("windowsize")[1]
    results_df["Step"] = input_dir.name.split("_")[1].split("step")[1]

    results_df["Videos"] = len(prediction_files)

    cols = results_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("Step")))
    cols.insert(0, cols.pop(cols.index("Window")))
    results_df = results_df[cols]

    print(
        results_df.to_latex(
            index=False,
            formatters={"name": str.upper},
            float_format="{:.2f}".format,
        )
    )


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
