import pandas as pd
from pathlib import Path

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
    "You make me sad.",
    "On second thoughts, let's not go to Camelot.",
    "It is a silly place.",
    "We want a shrubbery!",
    "I fart in your general direction!",
    "The Castle Aaargh!",
    "Found them? In Mercia?",
    "This is supposed to be a happy occasion!",
    "You must cut down the mightiest tree in the forest with... a herring!",
]


def generate_queries(
    video_id: str = "video_01",
    total_frames: int = 1000,
    num_segments: int = 20,
    output_path: Path = Path("example_queries.csv"),
):
    assert total_frames >= num_segments, "Total frames must be at least equal to number of segments"

    segment_length = total_frames // num_segments
    quotes = (MONTY_QUOTES * ((num_segments + len(MONTY_QUOTES) - 1) // len(MONTY_QUOTES)))[:num_segments]

    data = []
    for idx in range(num_segments):
        start_frame = idx * segment_length
        end_frame = start_frame + segment_length
        query_text = quotes[idx]
        data.append(
            {
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
    print(f"Wrote {num_segments} queries to {output_path.resolve()}")


if __name__ == "__main__":
    generate_queries()
