import argparse
import pandas as pd
import random
from pathlib import Path


def generate_predictions(df: pd.DataFrame, k: int, mode: str):
    grouped = df.groupby("query_text")
    rows = []

    all_seg_ids = df["seg_idx"].unique().tolist()

    for query, group in grouped:
        relevant = group["seg_idx"].tolist()
        video_id = group["video_id"].iloc[0]
        non_relevant = [idx for idx in all_seg_ids if idx not in relevant]

        if mode == "random":
            candidates = all_seg_ids.copy()
            random.shuffle(candidates)
            ranked = candidates[:k]

        elif mode == "perfect":
            others = [idx for idx in all_seg_ids if idx not in relevant]
            ranked = relevant + others
            ranked = ranked[:k]

        elif mode == "wrong":
            others = [idx for idx in all_seg_ids if idx not in relevant]
            ranked = others + relevant
            ranked = ranked[:k]

        elif mode == "half_right":
            random.shuffle(relevant)
            random.shuffle(non_relevant)
            ranked = []
            while len(ranked) < k and (relevant or non_relevant):
                if len(ranked) % 2 == 0 and relevant:
                    ranked.append(relevant.pop())
                elif non_relevant:
                    ranked.append(non_relevant.pop())
            ranked = ranked[:k]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        for rank, seg_idx in enumerate(ranked):
            rows.append(
                {
                    "video_id": video_id,
                    "query_text": query,
                    "rank": rank,
                    "seg_idx": seg_idx,
                }
            )

    return pd.DataFrame(rows)


def main():
    mode_choices = ["random", "perfect", "wrong", "half_right"]
    parser = argparse.ArgumentParser(description="Generate fake predictions using known answers")
    parser.add_argument("input_csv", type=Path, help="Path to input CSV")
    parser.add_argument("--k", type=int, default=10, help="Number of predictions per query")
    parser.add_argument("--mode", choices=mode_choices)
    parser.add_argument("--output-path", type=Path, default="example_predictions", help="Output CSV directory")

    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    if args.mode is None:
        modes = mode_choices
    else:
        modes = [args.mode]

    df = pd.read_csv(args.input_csv)

    required_cols = {"seg_idx", "query_text", "video_id", "start_frame", "end_frame"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    for mode in modes:
        output_csv = args.output_path / f"{mode}_predictions.csv"
        predictions_df = generate_predictions(df, args.k, mode)
        predictions_df.to_csv(output_csv, index=False)
        print(f"Saved {mode} predictions to {output_csv}")


if __name__ == "__main__":
    main()
