import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_random_predictions(row: pd.Series, num_predictions: int = 10) -> list[dict]:
    """Generate random (start_frame, end_frame) predictions for a given row."""
    predictions = []
    total_frames = int(row["total_frames"])

    for _ in range(num_predictions):
        # Randomly pick two frames and sort them to ensure start < end
        start, end = np.sort(np.random.randint(0, total_frames, size=2))
        # If start == end, force end_frame to be at least start + 1 within bounds
        if start == end:
            end = min(start + 1, total_frames)
        predictions.append(
            {
                "query_text": row["query_text"],
                "video_id": row["video_id"],
                "start_frame": int(start),
                "end_frame": int(end),
                "total_frames": total_frames,
            }
        )
    return predictions


def main(input_path: Path, output_path: Path = Path("predictions.csv")):
    df = pd.read_csv(input_path)
    all_predictions = []

    for _, row in df.iterrows():
        preds = generate_random_predictions(row, num_predictions=10)
        all_predictions.extend(preds)

    output_df = pd.DataFrame(all_predictions)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(output_df)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random frame predictions for queries.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default="example_queries.csv",
        help="Path to the input CSV file (default: example_queries.csv)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="random_predictions.csv",
        help="Path to the output CSV file (default: random_predictions.csv)",
    )
    args = parser.parse_args()

    main(input_path=args.input_path, output_path=args.output_path)
