import argparse
import random
from pathlib import Path

import pandas as pd

MONTY_QUOTES = [
    "Nobody expects the Spanish Inquisition!",
    "I'm not dead yet!",
    "It's just a flesh wound.",
    "This parrot is no more!",
    "What is the airspeed velocity of an unladen swallow?",
    "Your mother was a hamster!",
    "Now go away or I shall taunt you a second time!",
    "We are the knights who say... Ni!",
    "Strange women lying in ponds distributing swords is no basis for a system of government!",
    "Always look on the bright side of life.",
    "Bring out your dead!",
    "It is a silly place.",
    "We want a shrubbery!",
    "I fart in your general direction!",
    "The Castle Aaargh!",
    "You must cut down the mightiest tree in the forest with... a herring!",
]


def generate_queries(
    video_id_count: int,
    num_segments: int,
    total_frames: int,
    output_path: Path,
):
    assert total_frames >= num_segments, "Total frames must be at least equal to number of segments"

    segment_length = total_frames // num_segments

    # Repeat quotes list enough times and shuffle
    quotes = (MONTY_QUOTES * ((num_segments + len(MONTY_QUOTES) - 1) // len(MONTY_QUOTES)))[:num_segments]
    random.shuffle(quotes)
    quotes = random.choices(population=MONTY_QUOTES, k=num_segments)

    data = []
    for vid_idx in range(video_id_count):
        video_id = f"video_{vid_idx + 1:02d}"
        for seg_idx in range(num_segments):
            start_frame = seg_idx * segment_length
            end_frame = start_frame + segment_length
            query_text = quotes[seg_idx]
            data.append(
                {
                    "seg_idx": seg_idx,
                    "query_text": query_text,
                    "video_id": video_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "total_frames": total_frames,
                }
            )

    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} queries across {video_id_count} video_ids to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate non-overlapping query segments across multiple video IDs.")
    parser.add_argument("--video-id-count", type=int, default=1, help="Number of unique video IDs (default: 1)")
    parser.add_argument("--num-segments", type=int, default=20, help="Number of segments per video (default: 20)")
    parser.add_argument("--total-frames", type=int, default=1000, help="Total frames per video (default: 1000)")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("example_queries.csv"),
        help="Output CSV path (default: example_queries.csv)",
    )

    args = parser.parse_args()

    generate_queries(
        video_id_count=args.video_id_count,
        num_segments=args.num_segments,
        total_frames=args.total_frames,
        output_path=args.output_path,
    )
