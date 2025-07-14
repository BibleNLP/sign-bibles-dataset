#!/usr/bin/env python3

import subprocess
import csv
from pathlib import Path
from tqdm import tqdm


def get_duration(path):
    """Returns duration in seconds as float, or None on error."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error with {path}: {e}")
        return None


def main(
    root_dir="/data/petabyte/cleong/data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads/",
    output_csv="video_durations.csv",
):
    video_files = list(Path(root_dir).rglob("*.mp4"))
    rows = []

    for path in tqdm(video_files, desc="Getting durations"):
        duration = get_duration(path)
        if duration is not None:
            rows.append((str(path), duration))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "duration_seconds"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
