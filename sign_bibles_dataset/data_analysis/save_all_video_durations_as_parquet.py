import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_duration(filepath: Path) -> float:
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
                str(filepath),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def collect_durations(root_dir: Path) -> pd.DataFrame:
    video_paths = list(root_dir.rglob("*.mp4"))
    data = []

    for path in tqdm(video_paths, desc=f"Scanning {root_dir}"):
        if "animation" in path.name.lower():
            continue
        duration = get_duration(path)
        if duration is not None:
            data.append(
                {"file_path": str(path.resolve()), "folder": str(path.parent.resolve()), "duration_sec": duration}
            )

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Root directory to search for videos")
    parser.add_argument("output_file", type=Path, help="Path to output Parquet file")
    args = parser.parse_args()

    df = collect_durations(args.input_dir)
    df.to_parquet(args.output_file, index=False)
    print(f"Saved {len(df)} durations to {args.output_file}")


if __name__ == "__main__":
    main()
