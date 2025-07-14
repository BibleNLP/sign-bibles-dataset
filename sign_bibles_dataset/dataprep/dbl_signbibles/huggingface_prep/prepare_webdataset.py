#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
* Downloads videos from DBL-sign
* Packages everything into WebDataset format


# TODO: Rename files to include language code and project name, to ensure unique IDs
# TODO: Add autosegmenter .eaf file instead, though maybe clean them up to remove paths. (need pympi-ling)
# TODO: size-based sharding, try for not too big?
# TODO: Load in .ocr.manualedit.withvrefs.csv if available,
#   * if available add in transcripts with frame indices, biblenlp-vref, text.
#   * if not available add in one for the whole video
# TODO: read more of the information directly from project metadata.xml, e.g. rights holders
# TODO: rename mediapipe ".pose" files to ".pose-mediapipe.pose" as they are added.
# TODO: parse metadata.xml to get country code
# TODO: add in "glosses" to match https://huggingface.co/datasets/bridgeconn/sign-bibles-isl?
# example:
"glosses": [
        {
            "text": [
                [
                    0,
                    0,
                    "nil"
                ]
            ],
            "language": {
                "name": "English",
                "ISO639-3": "eng",
                "BCP-47": "en-US"
            }
        }
    ]

"""

import argparse
import io
import json
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import langcodes
import pandas as pd
from processing_logger import ProcessingLogger
from tqdm import tqdm

from sign_bibles_dataset.dataprep.dbl_signbibles.ebible_utils.vref_lookup import (
    citation_to_text_and_vrefs,
    load_bible_lines,
    load_vref_map,
)
from sign_bibles_dataset.dataprep.dbl_signbibles.huggingface_prep.dbl_sign_downloader import DBLSignDownloader

# Initialize the logger with the correct path
logger = ProcessingLogger(log_file_path="./output/run_log.txt")
# Clear the log at the start of the process
logger.clear_log()

# Add command line info to the log
logger.log_info(f"Command: {' '.join(sys.argv)}")


class WebDatasetCreator:
    """Class to handle creating WebDataset format."""

    def __init__(self, output_dir: str | Path = "webdataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_webdataset(self, samples_info: list[dict[str, Any]], shard_size: int = 1000) -> list[str]:
        shards = self._split_into_shards(samples_info, shard_size)
        return self._write_shards(shards)

    def _split_into_shards(self, samples_info: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
        shards = []
        for i in tqdm(range(0, len(samples_info), shard_size), desc="Sharding samples"):
            shards.append(samples_info[i : i + shard_size])
        return shards

    def _write_shards(self, shards: list[list[dict[str, Any]]]) -> list[str]:
        shard_paths = []

        for shard_index, shard in enumerate(tqdm(shards, desc="Writing Shards")):
            shard_path = self.output_dir / f"shard_{shard_index:05d}.tar"
            with tarfile.open(shard_path, "w") as tar:
                for video_info in shard:
                    sample = self._build_sample(video_info, shard_index)
                    if not sample:
                        continue
                    self._add_sample_to_tar(tar, sample, sample["__key__"])
            shard_paths.append(str(shard_path))

        return shard_paths

    def _build_sample(self, video_info: dict[str, Any], shard_index: int) -> dict[str, Any]:
        sample = {}

        # Prepare sample name
        metadata = video_info.copy()
        language_code = metadata["transcripts"][0]["language"]["ISO639-3"]
        project_name = metadata["project_name"]
        project_slug = self._slugify(project_name)
        original_name = Path(metadata["filename"]).stem

        sample_name = f"{language_code}_{project_slug}_{original_name}"
        sample["__key__"] = sample_name

        # Actual disk files
        video_path = Path(metadata["filename_path"])

        # Files inside tar: renamed
        sample["files"] = {f"{sample_name}.mp4": video_path}

        # Optional pose files
        # add metadata information for each of the pose files
        metadata["pose"] = {}
        dw_pose = video_path.with_suffix(".pose-dwpose.npz")
        if dw_pose.exists():
            desired_name = f"{sample_name}.pose-dwpose.npz"
            metadata["pose"]["dwpose"] = desired_name
            sample["files"][desired_name] = dw_pose

        mediapipe_pose = video_path.with_suffix(".pose")
        if mediapipe_pose.exists():
            desired_name = f"{sample_name}.pose-mediapipe.pose"
            metadata["pose"]["mediapipe"] = desired_name
            sample["files"][desired_name] = mediapipe_pose

        # don't actually add the paths to the final json
        del metadata["path"]
        del metadata["filename_path"]

        # overwrite filename
        metadata["filename"] = f"{sample_name}.mp4"

        # JSON metadata
        json_data = json.dumps(metadata)
        sample["json_data"] = json_data
        sample["json_filename"] = f"{sample_name}.json"

        return sample

    def _add_sample_to_tar(self, tar: tarfile.TarFile, sample: dict[str, Any], sample_name: str):
        # Add JSON metadata
        encoded = sample["json_data"].encode("utf-8")
        info = tarfile.TarInfo(sample["json_filename"])
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))

        # Add actual files
        for tar_filename, src_path in sample["files"].items():
            src_path = Path(src_path)
            if not src_path.is_file():
                print(f"Warning: File {src_path} not found, skipping.")
                continue
            info = tarfile.TarInfo(tar_filename)
            info.size = src_path.stat().st_size
            with src_path.open("rb") as f:
                tar.addfile(info, f)

    def _slugify(self, text: str) -> str:
        """Simplify project name into safe filename slug."""
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "_", text)
        return text.strip("_")


def parse_metadata_to_bible_ref(xml_path: Path, filename):
    """
    Parse a metadata.xml and return bible ref associated, or empty string
    e.g. "CBT-001-esl-3_Bible_God Creates Everything.mp4" -> "GEN 1:1-31,2:1-3"
    """
    print(f"Parsing metadata to find bible refs for {filename}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # dbl_id = root.attrib.get("id", None)
        file_to_passage = {}

        for pub in root.findall("./publications/publication"):
            for division in pub.findall("./structure/division"):
                passage = division.attrib.get("role", "").strip()
                for content in division.findall("content"):
                    src = content.attrib.get("src", "").strip()
                    xml_filename = Path(src).name
                    file_to_passage[xml_filename] = passage
        # print(json.dumps(file_to_passage, indent=2))
        passage_found = file_to_passage.get(filename, "")
        print(f"Found Passage {passage_found} for filename {filename}")
        return passage_found

    except ET.ParseError as e:
        print(f"[ERROR] Could not parse {xml_path}: {e}")
        return None, {}


def search_ebible_translations(
    language_code: str, translation_id: str, ebible_translations_df: pd.DataFrame
) -> dict | None:
    result = ebible_translations_df[
        (ebible_translations_df["languageCode"] == language_code)
        & (ebible_translations_df["translationId"] == translation_id)
    ]
    if result.empty:
        print(f"No results found for ({language_code}, {translation_id})")
        return None
    elif len(result) > 1:
        print(f"Warning: Multiple results found for ({language_code}, {translation_id}), returning the first.")

    return result.iloc[0].to_dict()


def get_video_info_with_opencv(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_opencv_info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        if cap.get(cv2.CAP_PROP_FPS) > 0
        else None,
    }

    cap.release()
    return video_opencv_info


def process_without_gui(args):
    """Process videos without GUI updates using pathlib for filesystem operations."""

    # === Step 0: Initialize Paths ===
    output_dir = Path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    webdataset_dir = output_dir / "webdataset"
    manifest_path = output_dir / "manifest.json"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    webdataset_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Load eBible Corpus ===
    ebible_corpus_path = args.ebible_corpus_path
    if not ebible_corpus_path.is_dir():
        raise ValueError(
            f"eBible Corpus not found at {ebible_corpus_path}. Please clone it there or pass --ebible-corpus-path"
        )

    vref_map = load_vref_map(ebible_corpus_path / "metadata" / "vref.txt")
    bible_verses = load_bible_lines(ebible_corpus_path / "corpus" / f"{args.ebible_version}.txt")
    ebible_translations_df = pd.read_csv(ebible_corpus_path / "metadata" / "translations.csv")

    print(f"Loaded {len(vref_map)} vrefs, {len(bible_verses)} verses, {len(ebible_translations_df)} translations")

    language_code, translation_id = args.ebible_version.split("-")
    ebible_version_metadata = search_ebible_translations(language_code, translation_id, ebible_translations_df)

    # === Step 2: Download Videos ===
    print(f"=== Downloading {args.num_videos} videos (language code: {args.language_code}) ===")
    downloader = DBLSignDownloader(downloads_dir)
    video_info_list = downloader.download_videos(args.num_videos, args.language_code, args.project_name)
    print(f"Downloaded {len(video_info_list)} videos")

    # === Step 3: Enrich Video Metadata with eBible References ===
    enriched_video_info_list = []
    for video_info in video_info_list:
        video_path = Path(video_info["path"])
        video_info["filename"] = video_path.name
        video_info["filename_path"] = str(video_path)

        video_info.update(get_video_info_with_opencv(video_path))

        meta_xml = video_path.parent / "metadata.xml"
        video_info["bible-ref"] = parse_metadata_to_bible_ref(meta_xml, video_path.name)

        # Add transcript from vref map and Bible text
        bible_text, vrefs = citation_to_text_and_vrefs(video_info["bible-ref"], vref_map, bible_verses)
        video_info["biblenlp-vref"] = vrefs
        # TODO: if there is a file ending with .ocr.manualedit.withrefs.csv, use it to add transcripts
        # if there are no more finegrained timestamps, make a single transcript spanning the whole video
        transcript = {
            "text": bible_text,
            "biblenlp-vref": vrefs,
            "bible-ref": video_info["bible-ref"],
            "start_frame": 0,
            "end_frame": video_info["total_frames"] - 1,
            "language": {
                "name": ebible_version_metadata["languageName"],
                "ISO639-3": ebible_version_metadata["languageCode"],
                "BCP-47": langcodes.standardize_tag(ebible_version_metadata["languageCode"]),
            },
            "license": ebible_version_metadata["Copyright"],
            "source": ebible_version_metadata["publicationURL"],
        }

        video_info["transcripts"] = [transcript]

        enriched_video_info_list.append(video_info)

    # === Step 4: Create WebDataset ===
    print(f"=== Creating WebDataset with {len(enriched_video_info_list)} samples ===")
    creator = WebDatasetCreator(webdataset_dir)
    shard_paths = creator.create_webdataset(enriched_video_info_list, shard_size=args.shard_size)

    print(f"Created {len(shard_paths)} shards in {webdataset_dir.resolve()}")

    # === Step 5: Save Manifest ===
    print("=== Saving Manifest ===")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"samples": enriched_video_info_list}, f, indent=2)

    print(f"Processing complete! Manifest saved to {manifest_path.resolve()}")
    logger.log(f"Done, see {logger.log_file_path} for logs")


def main():
    """Main function to run the entire process."""
    parser = argparse.ArgumentParser(description="Prepare sign language videos for HuggingFace datasets")
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to download")
    parser.add_argument(
        "--language-code",
        action="append",
        help="Filter by language code. You can specify this flag multiple times, e.g. --language-code eng --language-code spa",
    )
    parser.add_argument("--project-name", type=str, help="Filter by project name")
    parser.add_argument(
        "--auto-segment",
        type=str,
        help="Use autosegmentation from sign-language-processing",
    )
    parser.add_argument(
        "--ebible-corpus-path",
        type=Path,
        help="Root directory of the eBible corpus. https://github.com/BibleNLP/ebible",
        default=Path(Path(__file__).resolve().parent.parent) / "ebible/",
    )
    parser.add_argument(
        "--ebible-version",
        type=str,
        help="Which Bible from the eBible corpus to use for verse texts",
        default="eng-engbsb",
    )

    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10,
        help="Number of samples per WebDataset shard",
    )
    args = parser.parse_args()

    process_without_gui(args)


if __name__ == "__main__":
    main()
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/prepare_webdataset.py --output-dir . --language-code esl
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/prepare_webdataset.py --output-dir . --language-code sqs --language-code esl --num-videos 10000000000000
