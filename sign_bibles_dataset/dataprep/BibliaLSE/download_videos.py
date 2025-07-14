#!/usr/bin/env python3

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from pytubefix import YouTube
from pytubefix.exceptions import BotDetection, RegexMatchError, VideoUnavailable
from tqdm import tqdm


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download YouTube videos and metadata.")
    parser.add_argument("json_path", type=Path, help="Path to JSON file with video metadata")
    parser.add_argument("download_dir", type=Path, help="Directory to save downloaded videos")
    return parser.parse_args()


def load_metadata(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def download_video(video_id: str, metadata: dict[str, Any], base_dir: Path) -> None:
    """Download a YouTube video and save metadata if missing."""
    output_dir = base_dir / f"manos3d_{video_id}"
    mp4_files = list(output_dir.glob("*.mp4"))
    json_files = list(output_dir.glob("*.json"))

    if output_dir.exists():
        if mp4_files and json_files:
            logging.info(f"[{video_id}] Already downloaded and has metadata, skipping.")
            return
        elif mp4_files and not json_files:
            # Save metadata next to existing video
            logging.info(f"[{video_id}] MP4 found but no JSON. Saving metadata.")
            metadata_path = mp4_files[0].with_suffix(".json")
            try:
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except OSError as e:
                logging.error(f"[{video_id}] Failed to write metadata: {e}")
            return
        elif json_files and not mp4_files:
            logging.info(f"[{video_id}] JSON found but no mp4. Downloading video")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        watch_url = metadata["watch_url"]
    except KeyError:
        logging.warning(f"[{video_id}] Missing 'watch_url' in metadata.")
        return

    try:
        yt = YouTube(watch_url)
    except VideoUnavailable:
        logging.warning(f"[{video_id}] Video unavailable: {watch_url}")
        return
    except RegexMatchError:
        logging.warning(f"[{video_id}] Malformed YouTube URL: {watch_url}")
        return

    stream = yt.streams.filter(file_extension="mp4", progressive=False).get_highest_resolution()
    if not stream:
        logging.warning(f"[{video_id}] No MP4 stream found.")
        return

    try:
        downloaded_path = Path(stream.download(output_path=output_dir))
    except OSError as e:
        logging.error(f"[{video_id}] Download failed: {e}")
        return

    metadata_path = downloaded_path.with_suffix(".json")
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logging.error(f"[{video_id}] Failed to write metadata: {e}")
        return

    sleep_time = random.uniform(5.0, 30.0)
    logging.info(
        f"[{video_id}] Downloaded successfully to {downloaded_path}.\n Wrote metadata to {metadata_path}.\n Sleeping for {sleep_time:.1f} seconds."
    )
    time.sleep(sleep_time)


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.json_path.is_file():
        logging.error(f"Metadata JSON file not found: {args.json_path}")
        return

    args.download_dir.mkdir(parents=True, exist_ok=True)
    metadata_dict = load_metadata(args.json_path)

    botdetected_video_ids = []
    for video_id, metadata in tqdm(metadata_dict.items(), desc="Downloading videos"):
        try:
            download_video(video_id, metadata, args.download_dir)
        except BotDetection as e:
            logging.info(f"[{video_id}] cannot be downloaded: {type(e)} {e}.")
            botdetected_video_ids.append(video_id)
    print(f"{len(botdetected_video_ids)} could not be downloaded due to bot detection")
    for video_id in botdetected_video_ids:
        print(video_id)
        print(args.download_dir / f"manos3d_{video_id}")


if __name__ == "__main__":
    main()
# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/BibliaLSE/download_videos.py /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/BibliaLSE/manos_metadata.json /data/petabyte/cleong/data/biblia_lse/videos
