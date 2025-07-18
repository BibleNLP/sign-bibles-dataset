#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
* Downloads videos from DBL-sign
* Packages everything into WebDataset format
* Puts each language in its own subfolder in the webdataset

# TODO: Add autosegmenter .eaf file instead, though maybe clean them up to remove paths. (need pympi-ling)
# TODO: size-based sharding, try for not too big?
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

from collections import Counter
import argparse
import io
import json
import logging
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import langcodes
import pandas as pd
from tqdm import tqdm

from sign_bibles_dataset.dataprep.dbl_signbibles.ebible_utils.vref_lookup import (
    citation_to_text_and_vrefs,
    load_bible_lines,
    load_vref_map,
)
from sign_bibles_dataset.dataprep.dbl_signbibles.huggingface_prep.dbl_sign_downloader import DBLSignDownloader

# Initialize the logger with the correct path


def setup_logger(log_file_path: str) -> logging.Logger:
    log_file = Path(log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ProcessingLogger")
    logger.setLevel(logging.DEBUG)  # Capture everything

    # Clear existing handlers (important if rerunning in notebooks or scripts)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (all messages)
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console Handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


LOG_FILE_PATH = Path("./output/run_log.txt")
logger = setup_logger(LOG_FILE_PATH.resolve())
logger.info(f"Command: {' '.join(sys.argv)}")


class WebDatasetCreator:
    """Class to handle creating WebDataset format with subfolders per language."""

    def __init__(self, output_dir: str | Path = "webdataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _group_by_project(self, samples_info):
        groups = {}
        for sample in samples_info:
            project_slug = self._slugify(sample["project_name"])
            groups.setdefault(project_slug, []).append(sample)
        return groups

    def create_webdataset(self, samples_info: list[dict[str, Any]], shard_size: int = 1000) -> list[str]:
        # Group by language_code first
        language_groups = self._group_by_language(samples_info)
        all_shard_paths = []

        for language_code, samples in language_groups.items():
            # Further group by project_name within language
            project_groups = self._group_by_project(samples)

            for project_slug, project_samples in project_groups.items():
                logger.info(
                    f"Processing language {language_code}, project {project_slug}, {len(project_samples)} samples"
                )
                shards = self._split_into_shards(project_samples, shard_size)

                project_output_dir = self.output_dir / language_code / project_slug
                project_output_dir.mkdir(parents=True, exist_ok=True)

                shard_paths = self._write_shards(shards, project_output_dir)
                all_shard_paths.extend(shard_paths)

        return all_shard_paths

    def _group_by_language(self, samples_info: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        language_groups = {}
        for sample in samples_info:
            language_code = sample["language"]["ISO639-3"]
            language_groups.setdefault(language_code, []).append(sample)
        return language_groups

    def _split_into_shards(self, samples_info: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
        shards = []
        for i in tqdm(range(0, len(samples_info), shard_size), desc="Sharding samples"):
            shards.append(samples_info[i : i + shard_size])
        return shards

    def _write_shards(self, shards: list[list[dict[str, Any]]], output_dir: Path) -> list[str]:
        shard_paths = []

        for shard_index, shard in enumerate(tqdm(shards, desc=f"Writing shards in {output_dir.name}")):
            shard_path = output_dir / f"shard_{shard_index:05d}.tar"
            with tarfile.open(shard_path, "w") as tar:
                for video_info in shard:
                    sample = self._build_sample(video_info, shard_index)
                    if not sample:
                        continue
                    self._add_sample_to_tar(tar, sample, sample["__key__"])
            shard_paths.append(str(shard_path))

        return shard_paths

    def _save_transcripts(self, transcripts: list, filename: str) -> Path:
        """Save transcripts to a temp location for tar writing."""
        temp_path = Path("/tmp") / filename
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False)
        return temp_path

    def _build_sample(self, video_info: dict[str, Any], shard_index: int) -> dict[str, Any]:
        sample = {}

        metadata = video_info.copy()
        language_code = metadata["language"]["ISO639-3"]
        project_name = metadata["project_name"]
        project_slug = self._slugify(project_name)
        original_name = Path(metadata["filename"]).stem
        sample_name = f"{language_code}_{project_slug}_{original_name}"
        sample["__key__"] = sample_name

        video_path = Path(metadata["filename_path"])
        sample["files"] = {f"{sample_name}.mp4": video_path}

        # Pose files
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

        # Remove heavy fields
        metadata.pop("biblenlp-vref", None)
        metadata.pop("text", None)

        # Extract transcripts
        transcripts = metadata.pop("transcripts_file", None)
        if transcripts:
            transcripts_filename = f"{sample_name}.transcripts.json"
            sample["files"][transcripts_filename] = self._save_transcripts(transcripts, transcripts_filename)
            # metadata["transcripts_file"] = transcripts_filename # don't need the filename in the metadata!

        # Clean up
        # metadata["filename"] = f"{sample_name}.mp4" # don't need filename in metadata!
        metadata.pop("path", None)
        metadata.pop("filename_path", None)
        metadata.pop("transcripts", None)  # don't store this in the metadata anymore
        metadata.pop("pose", None)  # don't need filenames in the metadata, it's in the sample
        metadata.pop("filename", None)  # don't need filenames in the metadata, it's in the sample

        # Store metadata
        json_data = json.dumps(metadata)
        sample["json_data"] = json_data
        sample["json_filename"] = f"{sample_name}.json"

        return sample

    def _add_sample_to_tar(self, tar: tarfile.TarFile, sample: dict[str, Any], sample_name: str):
        encoded = sample["json_data"].encode("utf-8")
        info = tarfile.TarInfo(sample["json_filename"])
        info.size = len(encoded)
        tar.addfile(info, io.BytesIO(encoded))

        for tar_filename, src_path in sample["files"].items():
            src_path = Path(src_path)
            if not src_path.is_file():
                logger.debug(f"Warning: File {src_path} not found, skipping.")
                continue
            info = tarfile.TarInfo(tar_filename)
            info.size = src_path.stat().st_size
            with src_path.open("rb") as f:
                tar.addfile(info, f)

    def _slugify(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "_", text)
        return text.strip("_")


def parse_metadata_to_bible_ref(xml_path: Path, filename):
    """
    Parse a metadata.xml and return bible ref associated, or empty string
    e.g. "CBT-001-esl-3_Bible_God Creates Everything.mp4" -> "GEN 1:1-31,2:1-3"
    """
    logger.debug(f"Parsing metadata to find bible refs for {filename}")
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

        passage_found = file_to_passage.get(filename, "")
        logger.debug(f"Found Passage {passage_found} for filename {filename}")
        return passage_found

    except ET.ParseError as e:
        logger.error(f"[ERROR] Could not parse {xml_path}: {e}")
        return None, {}


def parse_metadata_to_info(xml_path: Path):
    """
    Parse a metadata.xml file and extract language, countries, and rights holders.
    Returns a tuple: (language_dict, countries_list, rights_holders_list)
    """
    logger.debug(f"Parsing metadata for language, countries, and rights holders: {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # LANGUAGE
        language_node = root.find("./language")
        language = (
            {
                "iso": language_node.findtext("iso", default="").strip(),
                "name": language_node.findtext("name", default="").strip(),
                "nameLocal": language_node.findtext("nameLocal", default="").strip(),
            }
            if language_node is not None
            else {}
        )

        # COUNTRIES
        countries = []
        for country in root.findall("./countries/country"):
            countries.append(
                {
                    "iso": country.findtext("iso", default="").strip(),
                    "name": country.findtext("name", default="").strip(),
                    "nameLocal": country.findtext("nameLocal", default="").strip(),
                }
            )

        # RIGHTS HOLDERS
        rights_holders = []
        for rh in root.findall("./agencies/rightsHolder"):
            rights_holders.append(
                {
                    # "uid": rh.findtext("uid", default="").strip(),
                    "name": rh.findtext("name", default="").strip(),
                    "abbr": rh.findtext("abbr", default="").strip(),
                    "url": rh.findtext("url", default="").strip(),
                }
            )

        return language, countries, rights_holders

    except ET.ParseError as e:
        logger.error(f"[ERROR] Could not parse {xml_path}: {e}")
        return {}, [], []


def search_ebible_translations(
    language_code: str, translation_id: str, ebible_translations_df: pd.DataFrame
) -> dict | None:
    result = ebible_translations_df[
        (ebible_translations_df["languageCode"] == language_code)
        & (ebible_translations_df["translationId"] == translation_id)
    ]
    if result.empty:
        logger.debug(f"No results found for ({language_code}, {translation_id})")  # expected, many don't have it
        return None
    elif len(result) > 1:
        logger.warning(f"Warning: Multiple results found for ({language_code}, {translation_id}), returning the first.")

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


def build_transcripts(
    video_path: Path,
    bible_ref: str,
    vref_map: dict[str, int],
    bible_verses: list[str],
    ebible_version_metadata: dict[str, Any],
    total_frames: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    language_info = {
        "name": ebible_version_metadata["languageName"],
        "ISO639-3": ebible_version_metadata["languageCode"],
        "BCP-47": langcodes.standardize_tag(ebible_version_metadata["languageCode"]),
    }
    license_info = ebible_version_metadata["Copyright"]
    source_url = ebible_version_metadata["publicationURL"]

    transcripts = []
    ocr_csv = video_path.with_suffix(".ocr.manualedit.csv")

    if ocr_csv.is_file():
        logger.debug(f"Using fine-grained transcript from {ocr_csv.name}")
        df = pd.read_csv(ocr_csv)
        frame_indices = df["frame_index"].tolist()
        reference_texts = df["text"].fillna("").tolist()

        for idx, start_frame in enumerate(frame_indices):
            end_frame = frame_indices[idx + 1] - 1 if idx + 1 < len(frame_indices) else total_frames - 1
            reference = reference_texts[idx].strip()
            if not reference:
                continue

            verse_text, vref_indices = citation_to_text_and_vrefs(reference, vref_map, bible_verses)

            transcripts.append(
                {
                    "text": verse_text,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "language": language_info,
                    "license": license_info,
                    "source": source_url,
                    "biblenlp-vref": vref_indices,
                    "bible-ref": reference,
                }
            )

        # Optional: biblenlp-vref can be merged from all segments if needed
        biblenlp_vrefs = sorted({v for t in transcripts for v in t["biblenlp-vref"]})

    else:
        logger.debug(f"No fine-grained transcript for {video_path.name}, using fallback single transcript.")
        verse_text, vref_indices = citation_to_text_and_vrefs(bible_ref, vref_map, bible_verses)
        if not verse_text:
            logger.warning(f"No verses found for {video_path} reference {bible_ref}, returning empty lists")
            return [], vref_indices

        transcripts.append(
            {
                "text": verse_text,
                "start_frame": 0,
                "end_frame": total_frames - 1,
                "language": language_info,
                "license": license_info,
                "source": source_url,
                "biblenlp-vref": vref_indices,
                "bible-ref": bible_ref,
            }
        )
        biblenlp_vrefs = vref_indices

    return transcripts, biblenlp_vrefs


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

    logger.info(
        f"eBible Corpus: Loaded {len(vref_map)} vrefs, {len(bible_verses)} verses, {len(ebible_translations_df)} translations"
    )

    language_code, translation_id = args.ebible_version.split("-")
    ebible_version_metadata = search_ebible_translations(language_code, translation_id, ebible_translations_df)

    # === Step 2: Download Videos ===
    logger.info(f"=== Downloading {args.num_videos} videos (language code: {args.language_code}) ===")
    downloader = DBLSignDownloader(downloads_dir)
    video_info_list = downloader.download_videos(args.num_videos, args.language_code, args.project_name)
    logger.info(f"Downloaded {len(video_info_list)} videos")

    # === Step #: Parse
    for video_info in video_info_list:
        meta_xml_language, meta_xml_countries, meta_xml_rights_holders = parse_metadata_to_info(
            Path(video_info["path"]).parent / "metadata.xml"
        )
        video_info["language"]["name"] = meta_xml_language["name"]
        video_info["language"]["nameLocal"] = meta_xml_language["nameLocal"]
        video_info["language"]["ISO639-3"] = meta_xml_language["iso"]
        # possibly try meta_xml_countries["countries"][0]["iso"]?
        video_info["language"]["BCP-47"] = langcodes.standardize_tag(meta_xml_language["iso"])

        video_info["copyright"] = meta_xml_rights_holders

    # === Step 3: Enrich Video Metadata with eBible References ===
    enriched_video_info_list = []
    for video_info in video_info_list:
        video_path = Path(video_info["path"])
        video_info["filename"] = video_path.name
        video_info["filename_path"] = str(video_path)

        video_info.update(get_video_info_with_opencv(video_path))

        meta_xml = video_path.parent / "metadata.xml"
        video_info["bible-ref"] = parse_metadata_to_bible_ref(meta_xml, video_path.name)
        video_info["transcripts"], video_info["biblenlp-vref"] = build_transcripts(
            video_path=video_path,
            bible_ref=video_info["bible-ref"],
            vref_map=vref_map,
            bible_verses=bible_verses,
            ebible_version_metadata=ebible_version_metadata,
            total_frames=video_info["total_frames"],
        )

        enriched_video_info_list.append(video_info)

    # === Step 4: Create WebDataset ===
    logger.info(f"=== Creating WebDataset with {len(enriched_video_info_list)} samples ===")
    creator = WebDatasetCreator(webdataset_dir)
    shard_paths = creator.create_webdataset(enriched_video_info_list, shard_size=args.shard_size)
    # Get parent folders for all shards
    parent_folders = [str(Path(shard_path).parent.resolve()) for shard_path in shard_paths]

    # Count occurrences
    folder_counts = Counter(parent_folders)

    logger.info(f"Created {len(shard_paths)} shards in {webdataset_dir.resolve()}, including the following subfolders:")
    for folder, count in folder_counts.items():
        logger.info(f"* {folder} — {count} shard(s)")

    # === Step 5: Save Manifest ===
    logger.info("=== Saving Manifest ===")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"samples": enriched_video_info_list}, f, indent=2)

    logger.info(f"Processing complete! Manifest saved to {manifest_path.resolve()}")
    logger.info(f"Done, see {LOG_FILE_PATH.resolve()} for logs")


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
        default=5,  # we were getting some big ones
        help="Number of samples per WebDataset shard",
    )
    args = parser.parse_args()

    process_without_gui(args)


if __name__ == "__main__":
    main()
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/prepare_webdataset.py --output-dir . --language-code esl
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && conda activate /opt/home/cleong/envs/sign-bibles-dataset && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/huggingface-prep/prepare_webdataset.py --output-dir . --language-code sqs --language-code esl --num-videos 10000000000000


# upload a project
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/prepare_webdataset.py --output-dir . --project-name "Chronological Bible Translation in American Sign Language (119 Introductions and Passages)" --num-videos 5000000
