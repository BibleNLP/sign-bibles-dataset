#!/usr/bin/env python3
"""
Prepare sign language videos for HuggingFace datasets.
This script:
* Downloads videos from DBL-sign
* Packages everything into WebDataset format

# TODO: Add autosegmenter .eaf file instead, though maybe clean them up to remove paths. (need pympi-ling)
# TODO: move the classes to their own files.
# TODO: import citation_to_text_and_vrefs from the actual ebible_utils instead of copying it in here
# TODO: size-based sharding, try for not too big?
# TODO: Load in .ocr.manualedit.withvrefs.csv if available,
#   * if available add in transcripts with frame indices, biblenlp-vref, text.
#   * if not available add in one for the whole video

# TODO: read more of the information directly from project metadata.xml, e.g. rights holders
# TODO: rename mediapipe ".pose" files to ".pose-mediapipe.pose" as they are added.
# TODO: don't add pose animations, they are big and easily generated.
# TODO: add in "glosses" to match https://huggingface.co/datasets/bridgeconn/sign-bibles-isl,
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
import os
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import langcodes
import pandas as pd
import requests
from processing_logger import ProcessingLogger
from tqdm import tqdm

from sign_bibles_dataset.dataprep.dbl_signbibles.dbl_sign import dbl_manifest_generator

# Initialize the logger with the correct path
logger = ProcessingLogger(log_file_path="./output/run_log.txt")
# Clear the log at the start of the process
logger.clear_log()

# Add command line info to the log
logger.log_info(f"Command: {' '.join(sys.argv)}")


PARENT_DIR = Path(__file__).resolve().parent.parent
DBL_SIGN_DIR = PARENT_DIR / "dbl_sign"


class DBLSignDownloader:
    """Class to handle downloading videos from DBL-sign."""

    def __init__(self, output_dir: Path | str):
        """Initialize the downloader with output directory."""
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_path = DBL_SIGN_DIR / "manifest.json"
        self.manifest = None

        # Load manifest
        self.load_manifest()

    def generate_fresh_manifest(self):
        """Generate a fresh manifest using the imported manifest generator functions."""
        print("Generating fresh manifest...")

        # Response data containing the list of sign language projects (hardcoded in the original script)
        response_data = dbl_manifest_generator.get_response_data()

        # Create manifest using the imported function
        manifest = dbl_manifest_generator.create_manifest(response_data)

        # Save manifest using the imported function
        dbl_manifest_generator.save_manifest(manifest, self.manifest_path)

        print("Fresh manifest generated successfully.")

        # Load the newly generated manifest
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        return self.manifest

    def load_manifest(self):
        """Load the manifest file."""
        if not self.manifest_path.exists():
            print("Manifest file not found. Generating a fresh one...")
            return self.generate_fresh_manifest()

        # Check if manifest is older than 24 hours
        manifest_age = time.time() - self.manifest_path.stat().st_mtime
        if manifest_age > 86400:  # 24 hours in seconds
            print("Manifest is older than 24 hours. Generating a fresh one...")
            return self.generate_fresh_manifest()

        print("Loading existing manifest...")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        return self.manifest

    def refresh_manifest(self):
        """Refresh the manifest using the manifest generator script."""
        try:
            self.generate_fresh_manifest()
        except Exception as e:
            raise Exception(f"Failed to refresh manifest: {e}") from None

    def download_videos(self, num_videos, language_codes=None, project_name=None):
        """
        Download a specified number of videos from DBL-sign.

        Args:
            num_videos: Number of videos to download
            language_codes: Optional filter by language codes
            project_name: Optional filter by project name

        Returns:
            List of dictionaries with video information

        """
        if "languages" not in self.manifest:
            raise ValueError("Invalid manifest structure: 'languages' key not found")

        if language_codes is None:
            language_codes = [language_codes]
        all_downloaded_videos = []
        for language_code in language_codes:
            # Filter projects based on criteria
            filtered_projects = []
            total_languages = len(self.manifest["languages"])
            print(f"Total Languages in Manifest: {total_languages}")
            for i, (lang_code, projects) in enumerate(self.manifest["languages"].items()):
                if language_code and lang_code != language_code:
                    continue

                for proj_name, proj_info in projects.items():
                    # Only filter by project name if it's explicitly provided
                    if project_name and proj_name != project_name:
                        continue

                    # Count MP4 files in this project
                    mp4_count = sum(
                        1 for file_info in proj_info["files"] if file_info["filename"].lower().endswith(".mp4")
                    )

                    if mp4_count > 0:
                        filtered_projects.append((lang_code, proj_name, proj_info))

            if not filtered_projects:
                filter_criteria = f"language_code={language_code}"
                if project_name:
                    filter_criteria += f", project_name={project_name}"
                error_msg = f"No projects found matching the criteria ({filter_criteria})"

                raise ValueError(error_msg)

            # Download videos
            downloaded_videos = []
            videos_downloaded = 0

            print(f"Found {len(filtered_projects)} projects matching the criteria")

            for i, (lang_code, proj_name, proj_info) in enumerate(filtered_projects):
                if videos_downloaded >= num_videos:
                    break

                project_dir = self.output_dir / lang_code / proj_name
                print(f"Project Dir: {project_dir}")
                os.makedirs(project_dir, exist_ok=True)

                # Get metadata for this project
                metadata = {
                    "language": {
                        "ISO639-3": lang_code,
                        "BCP-47": langcodes.standardize_tag(lang_code),
                    },
                    "project_name": proj_name,
                    "copyright": proj_info.get("rights_holder", ""),
                    "license": proj_info.get("license", ""),
                    "source": proj_info.get("url", ""),
                }

                # Save metadata
                with open(project_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                # Download MP4 files
                mp4_files = [f for f in proj_info["files"] if f["filename"].lower().endswith(".mp4")]
                for j, file_info in enumerate(mp4_files):
                    if videos_downloaded >= num_videos:
                        break

                    filename = file_info["filename"]
                    filepath = Path(project_dir) / filename
                    url = file_info["download_url"]

                    # Skip if file already exists
                    if filepath.exists() and filepath.stat().st_size > 0:
                        # print(f"File already exists: {filepath}")

                        downloaded_videos.append({"path": str(filepath), **metadata})
                        videos_downloaded += 1
                        continue

                    # Download the file
                    print(f"Downloading {url} to {filepath}")

                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()

                        # Show progress bar
                        downloaded_size = 0
                        block_size = 8192

                        with open(filepath, "wb") as f:
                            for chunk in response.iter_content(chunk_size=block_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)

                        # Validate MP4

                        try:
                            cap = cv2.VideoCapture(filepath)
                            if not cap.isOpened():
                                error_msg = f"Warning: Could not open downloaded file as video: {filepath}"
                                print(error_msg)

                                continue
                            cap.release()

                        except Exception as e:
                            error_msg = f"Error validating video: {e}"
                            print(error_msg)

                            continue

                        downloaded_videos.append({"path": filepath, **metadata})
                        videos_downloaded += 1

                    except Exception as e:
                        error_msg = f"Error downloading {url}: {e}"
                        print(error_msg)

            print(f"Downloaded {videos_downloaded} videos")
            all_downloaded_videos.extend(downloaded_videos)
        return all_downloaded_videos


class WebDatasetCreator:
    """Class to handle creating WebDataset format."""

    def __init__(self, output_dir: str = "webdataset"):
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
                for sample_info in shard:
                    sample = self._build_sample(sample_info, shard_index)
                    if not sample:
                        continue
                    self._add_sample_to_tar(tar, sample, sample_info["sample_name"])
            shard_paths.append(str(shard_path))

        return shard_paths

    def _build_sample(self, sample_info: dict[str, Any], shard_index: int) -> dict[str, str | Path]:
        sample = {}

        sample["json"] = json.dumps(sample_info["sample_metadata"])
        sample["mp4"] = sample_info["sample_path"]
        for ext, path in sample_info["files_to_add"]:
            sample[ext] = path

        return sample

    def _add_sample_to_tar(
        self,
        tar: tarfile.TarFile,
        sample: dict[str, str | Path],
        sample_name: str,
    ):
        for ext, content in sample.items():
            filename = f"{sample_name}.{ext}"
            try:
                if ext == "json":
                    encoded = content.encode("utf-8")
                    info = tarfile.TarInfo(filename)
                    info.size = len(encoded)
                    tar.addfile(info, io.BytesIO(encoded))
                else:
                    content = Path(content)
                    size = content.stat().st_size
                    if size == 0:
                        print(f"Warning: File {content} has zero size. Skipping.")
                        continue
                    info = tarfile.TarInfo(filename)
                    info.size = size
                    with content.open("rb") as f:
                        tar.addfile(info, f)
            except Exception as e:
                print(f"Error adding {filename} to tar: {e}")


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


def load_vref_map(vref_path: str) -> dict[str, int]:
    with open(vref_path, encoding="utf-8") as f:
        return {line.strip(): idx for idx, line in enumerate(f) if line.strip()}


def load_bible_lines(bible_path: str) -> list[str]:
    with open(bible_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def parse_citation_string(citation: str, vref_map: dict[str, int]) -> list[int]:
    all_indices = []
    current_book = None
    current_chapter = None
    tokens = re.split(r";\s*", citation.strip())

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)\s+(\d+)", token)
        if m:
            book, chapter = m.groups()
            current_book = book
            current_chapter = chapter
            prefix = f"{book} {chapter}:"
            matches = [i for ref, i in vref_map.items() if ref.startswith(prefix)]
            all_indices.extend(matches)
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)", token)
        if m:
            book = m.group(1)
            current_book = book
            matches = [i for ref, i in vref_map.items() if ref.startswith(f"{book} ")]
            all_indices.extend(matches)
            continue

        m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+:\d+)\s*-\s*([1-3]?[A-Z]+)?\s*(\d+:\d+)", token)
        if m:
            book1, start, book2, end = m.groups()
            if book1:
                current_book = book1
            book2 = book2 or current_book
            if not (current_book and book2):
                continue
            start_ref = f"{current_book} {start}"
            end_ref = f"{book2} {end}"
            if start_ref in vref_map and end_ref in vref_map:
                i1, i2 = vref_map[start_ref], vref_map[end_ref]
                if i1 <= i2:
                    all_indices.extend(range(i1, i2 + 1))
            continue

        parts = token.split(",")
        for part in parts:
            part = part.strip()
            m = re.fullmatch(r"([1-3]?[A-Z]+)?\s*(\d+):(\d+(?:-\d+)?)", part)
            if m:
                maybe_book, ch, verse_range = m.groups()
                if maybe_book:
                    current_book = maybe_book
                current_chapter = ch
                if not current_book:
                    continue

                if "-" in verse_range:
                    start_v, end_v = map(int, verse_range.split("-"))
                    verse_numbers = range(start_v, end_v + 1)
                else:
                    verse_numbers = [int(verse_range)]

                for v in verse_numbers:
                    ref = f"{current_book} {current_chapter}:{v}"
                    if ref in vref_map:
                        all_indices.append(vref_map[ref])
    return sorted(set(all_indices))


def citation_to_text_and_vrefs(citation: str, vref_map, bible_verses):
    vrefs = parse_citation_string(citation, vref_map)

    verses = [bible_verses[i] for i in vrefs if 0 <= i < len(bible_verses)]

    bible_text = "".join(verses)
    return bible_text, vrefs


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
    # Initialize paths
    output_dir = Path(args.output_dir)
    downloads_dir = output_dir / "downloads"
    processed_dir = output_dir / "processed"
    webdataset_dir = output_dir / "webdataset"
    manifest_path = output_dir / "manifest.json"

    # eBible Corpus
    ebible_corpus_path = args.ebible_corpus_path
    vref_text_path = ebible_corpus_path / "metadata" / "vref.txt"
    ebible_translations_csv_path = ebible_corpus_path / "metadata" / "translations.csv"
    ebible_text_path = ebible_corpus_path / "corpus" / f"{args.ebible_version}.txt"
    if not ebible_corpus_path.is_dir():
        raise ValueError(
            f"eBible Corpus is not at {ebible_corpus_path}. Please clone it there or pass --ebible-corpus-path"
        )

    vref_map = load_vref_map(vref_text_path)
    bible_verses = load_bible_lines(ebible_text_path)
    print(f"Loading vref-to-index map from {vref_text_path}: {len(vref_map)} keys")
    print(f"Loading {args.ebible_version} verses from {ebible_text_path}: {len(bible_verses)} loaded")
    ebible_translations_df = pd.read_csv(ebible_translations_csv_path)
    print(f"Loading translations info from {ebible_translations_csv_path}: {len(ebible_translations_df)} translations")

    # Assert uniqueness of (languageCode, translationId)
    # duplicates = ebible_translations_df.duplicated(subset=["languageCode", "translationId"], keep=False)
    # assert not duplicates.any(), f"Duplicate rows found:\n{ebible_translations_df[duplicates]}"
    language_code, translation_id = args.ebible_version.split("-")
    ebible_version_metadata = search_ebible_translations(language_code, translation_id, ebible_translations_df)
    # print(f"Metadata for {args.ebible_version}: {ebible_version_metadata}")

    # Create directories
    downloads_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download videos
    print(f"=== Step 1: Downloading videos from DBL-sign. Language code filter: {args.language_code} ===")
    downloader = DBLSignDownloader(downloads_dir)

    # TODO: default num_videos only 10 even if already downloaded!
    video_info_list = downloader.download_videos(args.num_videos, args.language_code, args.project_name)
    print(f"Video Info List has {len(video_info_list)} items total")

    # print(json.dumps(video_info_list, indent=4))

    # TODO: get language name and iso code and country from metadata.xml, need a function
    #  also copyright statement
    # <copyright>
    #     <fullStatement>
    #       <statementContent type="xhtml">
    #         <p>© 2019-2021 Deaf Harbor</p>
    #         <p>© 2019-2021 D.O.O.R. International</p>
    #       </statementContent>
    #     </fullStatement>
    #   </copyright>
    # ---------------------------------------
    # Parse verse data from metadata.xml
    video_info_list_with_bible_refs = []
    for video_info in video_info_list:
        video_info["filename"] = Path(video_info["path"]).name

        video_opencv_info = get_video_info_with_opencv(video_info["path"])
        video_info.update(video_opencv_info)

        meta_xml = Path(video_info["path"]).parent / "metadata.xml"

        video_info["transcripts"] = []

        video_info["bible-ref"] = parse_metadata_to_bible_ref(meta_xml, Path(video_info["path"]).name)

        # look for .withvrefs.csv here and if so load transcripts from there
        transcript = {}
        bible_text, vrefs = citation_to_text_and_vrefs(
            video_info["bible-ref"], vref_map=vref_map, bible_verses=bible_verses
        )
        video_info["biblenlp-vref"] = vrefs
        transcript["text"] = bible_text

        # {'languageCode': 'eng', 'translationId': 'engbsb', 'languageName': 'English', 'languageNameInEnglish': 'English', 'dialect': nan, 'homeDomain': 'ebible.org', 'title': 'Berean Standard Bible', 'description': 'The Holy Bible in English: Berean Standard Bible', 'Redistributable': True, 'Copyright': 'public domain', 'UpdateDate': '2024-07-13', 'publicationURL': 'http://ebible.org/engbsb/', 'OTbooks': 39, 'OTchapters': 929, 'OTverses': 23145, 'NTbooks': 27, 'NTchapters': 260, 'NTverses': 7941, 'DCbooks': 0, 'DCchapters': 0, 'DCverses': 0, 'FCBHID': 'ENGBSB', 'Certified': True, 'inScript': 'http://eBible.org/study/?v1=GN1_1&w1=bible&t1=local%3A', 'swordName': 'engbsb2020eb', 'rodCode': nan, 'textDirection': 'ltr', 'downloadable': True, 'font': 'DejaVu Serif', 'shortTitle': 'English Berean Standard Bible', 'PODISBN': nan, 'script': 'Latin', 'sourceDate': '2024-07-13'}
        transcript["language"] = {
            "name": ebible_version_metadata["languageName"],
            "ISO639-3": ebible_version_metadata["languageCode"],
            "BCP-47": langcodes.standardize_tag(ebible_version_metadata["languageCode"]),
            "start_frame": 0,
            "end_frame": video_info["total_frames"],
        }
        transcript["license"] = ebible_version_metadata["Copyright"]
        transcript["source"] = ebible_version_metadata["publicationURL"]
        # transcript["source"] = ebible_version_metadata["title"]
        video_info["transcripts"].append(transcript)
        # ebible_version_metadata

        video_info_list_with_bible_refs.append(video_info)

    video_info_list = video_info_list_with_bible_refs

    print("=== Step 2: Processing videos to samples ===")
    all_samples = []
    for video_info in video_info_list:
        video_path = Path(video_info["path"])
        del video_info["path"]
        video_info["pose"] = {}

        files_to_add = []
        pose_animation_path = video_path.with_suffix(".pose-animation.mp4")
        if pose_animation_path.is_file():
            video_info["pose"]["animation"] = str(pose_animation_path.name)
            files_to_add.append(("pose-animation.mp4", str(pose_animation_path)))

        dw_pose_path = video_path.with_suffix(".pose-dwpose.npz")
        if dw_pose_path.is_file():
            video_info["pose"]["dwpose"] = str(dw_pose_path.name)
            files_to_add.append(("pose-dwpose.npz", str(dw_pose_path)))

        mediapipe_path = video_path.with_suffix(".pose")
        if mediapipe_path.is_file():
            video_info["pose"]["mediapipe"] = str(mediapipe_path.name)
            files_to_add.append(("pose", str(mediapipe_path)))

        sample_info = {
            "sample_name": video_path.stem,
            "sample_path": str(video_path),
            "sample_metadata": {
                **video_info,
            },
            "files_to_add": files_to_add,
        }

        all_samples.append(sample_info)

    # Step #: Create WebDatasetF
    print("=== Step 3: Creating WebDataset ===")
    webdataset_dir.mkdir(parents=True, exist_ok=True)
    print(webdataset_dir.resolve())

    creator = WebDatasetCreator(webdataset_dir)
    creator.create_webdataset(all_samples, shard_size=args.shard_size)

    # Step 4: Create manifest
    print("=== Step 4: Creating manifest ===")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"samples": all_samples}, f, indent=2)

    print(f"Processing complete! Manifest saved to {manifest_path.resolve()}")


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
        default=Path(PARENT_DIR) / "ebible/",
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
