import json
import argparse
from pathlib import Path

def get_total_duration(json_files):
    """Sum the duration field from all JSON files."""
    total_seconds = 0
    vrefs = []
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                duration = data.get('duration_sec')
                duration = float(duration)
                if isinstance(duration, (int, float)):
                    total_seconds += duration
                vrefs += data.get('biblenlp-vref', [])
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return total_seconds, len(set(vrefs))

def format_duration(seconds):
    """Convert total seconds into hours, minutes, and seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return int(hours), int(minutes), int(secs)

def main():
    parser = argparse.ArgumentParser(description='Calculate total duration from JSON files.')
    parser.add_argument('directories', nargs='+', help='List of directories to search')
    args = parser.parse_args()
    print(f"{args.directories=}")

    json_files = [
        video_file.with_suffix('.json') 
        for directory in args.directories
        for video_file in Path(directory).rglob("*.mp4") 
     ]
    print(f'Total number of videos : {len(json_files)}')
    total_seconds, total_verses = get_total_duration(json_files)
    print(f"{total_seconds=}")
    h, m, s = format_duration(total_seconds)
    print(f"Total duration: {h} hours, {m} minutes, {s} seconds")
    print(f"Total number of unique verses: {total_verses}")

if __name__ == '__main__':
    main()
