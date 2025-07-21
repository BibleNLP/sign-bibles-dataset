import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_random_predictions(row: pd.Series, num_predictions: int = 10) -> list[dict]:
    """Generate random (start_frame, end_frame) predictions for a given row."""
    predictions = []
    total_frames = int(row["total_frames"])

    for rank in range(1, num_predictions + 1):
        start, end = np.sort(np.random.randint(0, total_frames, size=2))
        if start == end:
            end = min(start + 1, total_frames)
        predictions.append(
            {
                "query_text": row["query_text"],
                "video_id": row["video_id"],
                "start_frame": int(start),
                "end_frame": int(end),
                "rank": rank,
            }
        )
    return predictions


def main(input_path: Path, output_path: Path = Path("random_predictions.csv"), num_predictions=10):
    df = pd.read_csv(input_path)
    all_predictions = []

    for _, row in df.iterrows():
        preds = generate_random_predictions(row, num_predictions=num_predictions)
        all_predictions.extend(preds)

    output_df = pd.DataFrame(all_predictions)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(output_df)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random frame predictions for queries.")
    parser.add_argument("--input-path", type=Path, default="example_queries.csv")
    parser.add_argument("--output-path", type=Path, default="random_predictions.csv")
    parser.add_argument("--num-predictions", type=int, default=10, help="Total number of predictions per query.")
    args = parser.parse_args()

    main(input_path=args.input_path, output_path=args.output_path)
