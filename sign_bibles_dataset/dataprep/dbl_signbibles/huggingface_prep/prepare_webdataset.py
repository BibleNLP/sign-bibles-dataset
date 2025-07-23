#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
* Downloads videos from DBL-sign
* Packages everything into WebDataset format
* Puts each language in its own subfolder in the webdataset

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

import argparse
import io
import json
import logging
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import langcodes
import pandas as pd
from sign_bibles_dataset.dataprep.dbl_signbibles.ebible_utils.vref_lookup import (
    citation_to_text_and_vrefs,
    load_bible_lines,
    load_vref_map,
)
from sign_bibles_dataset.dataprep.dbl_signbibles.huggingface_prep.dbl_sign_downloader import (
    DBLSignDownloader,
)
from sign_bibles_dataset.dataprep.dbl_signbibles.sign_segmentation.extract_eaf_to_json import (
    recursive_eaf_to_json,
)
from sign_bibles_dataset.dataprep.dbl_signbibles.sign_segmentation.recursively_run_segmentation import (
    MODEL_CHOICES,
)
from tqdm import tqdm


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

# PyArrow capacity limit (2^31 - 2)
PYARROW_MAX_BYTES = 2**31 - 2


class WebDatasetCreator:
    """Class to handle creating WebDataset format with subfolders per language, skipping samples with oversized files."""

    def __init__(self, output_dir: str | Path = "webdataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skipped_samples = []
        self.max_file_size_bytes = PYARROW_MAX_BYTES

    def _group_by_project(self, samples_info):
        groups = defaultdict(list)
        for sample in samples_info:
            project_slug = self._slugify(sample["project_name"])
            groups[project_slug].append(sample)
        return groups

    def create_webdataset(self, samples_info, shard_size=1000):
        language_groups = self._group_by_language(samples_info)
        all_shard_paths = []

        for language_code, samples in language_groups.items():
            project_groups = self._group_by_project(samples)

            for project_slug, project_samples in project_groups.items():
                logger.info(
                    f"Processing language {language_code}, project {project_slug}, {len(project_samples)} samples"
                )
                shards = self._split_into_shards(project_samples, shard_size)

                project_output_dir = self.output_dir / language_code / project_slug
                project_output_dir.mkdir(parents=True, exist_ok=True)

                for shard_index, shard in enumerate(
                    tqdm(shards, desc=f"Writing shards for {project_slug}")
                ):
                    shard_path = project_output_dir / f"shard_{shard_index:05d}.tar"
                    with tarfile.open(shard_path, "w") as tar:
                        for video_info in shard:
                            if self._is_sample_too_large(video_info):
                                continue
                            sample = self._build_sample(video_info, shard_index)
                            if sample:
                                self._add_sample_to_tar(tar, sample)
                    all_shard_paths.append(str(shard_path))

        if self.skipped_samples:
            logger.info(
                f"Skipped {len(self.skipped_samples)} samples due to large files:"
            )
            for entry in self.skipped_samples:
                logger.info(
                    f"- Sample: {entry['sample']} | File: {entry['large_file']} | Size: {entry['size_gb']:.2f} GB"
                )

        return all_shard_paths

    def _group_by_language(self, samples_info):
        language_groups = defaultdict(list)
        for sample in samples_info:
            language_code = sample["language"]["ISO639-3"]
            language_groups[language_code].append(sample)
        return language_groups

    def _split_into_shards(self, samples_info, shard_size):
        return [
            samples_info[i : i + shard_size]
            for i in range(0, len(samples_info), shard_size)
        ]

    def _save_transcripts(self, transcripts, filename):
        temp_path = Path("/tmp") / filename
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False)
        return temp_path

    def _is_sample_too_large(self, video_info):
        """Check if any file in this sample exceeds the max file size."""
        video_path = Path(video_info["filename_path"])
        files_to_check = [video_path]

        optional_suffixes = [
            ".eaf",
            ".autosegmented_segments.json",
            ".pose-dwpose.npz",
            ".pose",
        ]
        for suffix in optional_suffixes:
            path = video_path.with_suffix(suffix)
            if path.exists():
                files_to_check.append(path)

        for path in files_to_check:
            if path.stat().st_size > self.max_file_size_bytes:
                logger.warning(
                    f"Skipping sample {video_info.get('filename')} due to large file {path.name} ({path.stat().st_size / (1024**3):.2f} GB)"
                )
                self.skipped_samples.append(
                    {
                        "sample": video_info.get("filename"),
                        "large_file": path.name,
                        "size_gb": path.stat().st_size / (1024**3),
                    }
                )
                return True
        return False

    def _build_sample(self, video_info, shard_index):
        sample = {}
        metadata = video_info.copy()

        language_code = metadata["language"]["ISO639-3"].lower()
        project_slug = self._slugify(metadata["project_name"])
        original_name = Path(metadata["filename"]).stem.lower()
        sample_name = f"{language_code}_{project_slug}_{original_name}".lower()
        sample["__key__"] = sample_name

        video_path = Path(metadata["filename_path"])
        sample["files"] = {f"{sample_name}.mp4": video_path}

        suffixes = {
            # ".pose-dwpose.npz": ".pose-dwpose.npz", # TODO: add this back in once they're all done.
            ".pose": ".pose-mediapipe.pose",
        }

        for model_pth in MODEL_CHOICES:
            model_stem = Path(model_pth).stem  # .model_E4s-1, .model_E1s-1

            for suffix in [".eaf", ".autosegmented_segments.json"]:
                model_and_suffix = f".{model_stem}{suffix}"
                # turns out huggingface cannot handle uppercase filename extensions, so we lowercase them
                suffixes[model_and_suffix] = model_and_suffix.lower()

        for src_suffix, dest_suffix in suffixes.items():
            path = video_path.with_suffix(src_suffix)
            if path.exists():
                sample["files"][f"{sample_name}{dest_suffix}"] = path

        # empty list as default
        transcripts = metadata.pop("transcripts", [])
        transcripts_filename = f"{sample_name}.transcripts.json"
        transcripts_path = self._save_transcripts(transcripts, transcripts_filename)
        sample["files"][transcripts_filename] = transcripts_path
        if (
            len(transcripts) > 0
        ):  # TODO: check if "text" is in any item and item["text"] is not an empty string
            logger.debug(
                f"Shard {shard_index}, sample {sample_name}: we have transcripts with len {len(transcripts)}"
            )
        else:
            logger.debug(
                f"Shard {shard_index}, sample {sample_name}: NO transcripts: {transcripts}"
            )

        for key in [
            "biblenlp-vref",
            "text",
            "path",
            "filename_path",
            "transcripts_file",
            "transcripts",
            "pose",
            "filename",
        ]:
            metadata.pop(key, None)

        sample["json_data"] = json.dumps(metadata)
        sample["json_filename"] = f"{sample_name}.json"

        return sample

    def _add_sample_to_tar(self, tar, sample):
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
    logger.debug(
        f"Parsing metadata for language, countries, and rights holders: {xml_path}"
    )
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
        logger.debug(
            f"No results found for ({language_code}, {translation_id})"
        )  # expected, many don't have it
        return None
    elif len(result) > 1:
        logger.warning(
            f"Warning: Multiple results found for ({language_code}, {translation_id}), returning the first."
        )

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
            end_frame = (
                frame_indices[idx + 1] - 1
                if idx + 1 < len(frame_indices)
                else total_frames - 1
            )
            reference = reference_texts[idx].strip()
            if not reference:
                continue

            verse_text, vref_indices = citation_to_text_and_vrefs(
                reference, vref_map, bible_verses
            )

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
        logger.debug(
            f"No fine-grained transcript for {video_path.name}, using fallback single transcript."
        )
        verse_text, vref_indices = citation_to_text_and_vrefs(
            bible_ref, vref_map, bible_verses
        )
        if not verse_text:
            logger.warning(
                f"No verses found for {video_path} reference {bible_ref}, returning empty lists"
            )
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
    bible_verses = load_bible_lines(
        ebible_corpus_path / "corpus" / f"{args.ebible_version}.txt"
    )
    ebible_translations_df = pd.read_csv(
        ebible_corpus_path / "metadata" / "translations.csv"
    )

    logger.info(
        f"eBible Corpus: Loaded {len(vref_map)} vrefs, {len(bible_verses)} verses, {len(ebible_translations_df)} translations"
    )

    language_code, translation_id = args.ebible_version.split("-")
    ebible_version_metadata = search_ebible_translations(
        language_code, translation_id, ebible_translations_df
    )

    # === Step 2: Download Videos ===
    logger.info(
        f"=== Downloading {args.num_videos} videos (language code: {args.language_code}) ==="
    )
    downloader = DBLSignDownloader(downloads_dir)
    video_info_list = downloader.download_videos(
        args.num_videos, args.language_code, args.project_name
    )
    logger.info(f"Downloaded {len(video_info_list)} videos")

    # === Step #: Parse
    for video_info in video_info_list:
        meta_xml_language, meta_xml_countries, meta_xml_rights_holders = (
            parse_metadata_to_info(Path(video_info["path"]).parent / "metadata.xml")
        )
        video_info["language"]["name"] = meta_xml_language["name"]
        video_info["language"]["nameLocal"] = meta_xml_language["nameLocal"]
        video_info["language"]["ISO639-3"] = meta_xml_language["iso"]
        # possibly try meta_xml_countries["countries"][0]["iso"]?
        video_info["language"]["BCP-47"] = langcodes.standardize_tag(
            meta_xml_language["iso"]
        )

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

    # TODO: run pose estimators in here too

    # === Run autosegmenter and also convert the eafs to jsons for convenience
    parents = set()
    for video_info in video_info_list:
        parents.add(Path(video_info["filename_path"]).parent.resolve())

    for parent in parents:
        # slow, TODO: put this back in.
        # logger.info(f"Running autosegmenter on {parent}")
        # recursively_run_segmentation(parent)
        logger.info(f"Converting autosegmenter outputs for {parent} to json")
        recursive_eaf_to_json(parent)

    # === Step 4: Create WebDataset ===
    logger.info(
        f"=== Creating WebDataset with {len(enriched_video_info_list)} samples ==="
    )
    creator = WebDatasetCreator(webdataset_dir)
    shard_paths = creator.create_webdataset(
        enriched_video_info_list, shard_size=args.shard_size
    )
    # Get parent folders for all shards
    parent_folders = [
        str(Path(shard_path).parent.resolve()) for shard_path in shard_paths
    ]

    # Count occurrences
    folder_counts = Counter(parent_folders)

    logger.info(
        f"Created {len(shard_paths)} shards in {webdataset_dir.resolve()}, including the following subfolders:"
    )
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
    parser = argparse.ArgumentParser(
        description="Prepare sign language videos for HuggingFace datasets"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=10000000000000,
        help="Number of videos to download",
    )
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

    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
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

# do a number of projects
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/prepare_webdataset.py --output-dir . --language-code ase --language-code esl --language-code eso --language-code gse --language-code ins --language-code nsp --language-code sqs --num-videos 50000000
# Do those same ones but NOT ase
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/prepare_webdataset.py --output-dir . --language-code esl --language-code eso --language-code gse --language-code ins --language-code nsp --language-code sqs --num-videos 50000000


# upload a project
# cd "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset" && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/dataprep/dbl_signbibles/huggingface_prep/prepare_webdataset.py --output-dir . --project-name "Chronological Bible Translation in American Sign Language (119 Introductions and Passages)" --num-videos 5000000
