import xml.etree.ElementTree as ET
from pathlib import Path
import json
import argparse


def parse_metadata(xml_path: Path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        dbl_id = root.attrib.get("id", None)

        file_to_passage = {}
        for division in root.findall("./publications/publication/structure/division"):
            passage = division.attrib.get("role", "").strip()
            for content in division.findall("content"):
                src = content.attrib.get("src", "").strip()
                filename = Path(src).name
                file_to_passage[filename] = passage

        return dbl_id, file_to_passage
    except ET.ParseError as e:
        print(f"[ERROR] Could not parse {xml_path}: {e}")
        return None, {}


def collect_grouped_metadata(base_dir: Path):
    grouped = []

    for language_dir in base_dir.iterdir():
        if not language_dir.is_dir():
            continue
        language_code = language_dir.name

        for version_dir in language_dir.iterdir():
            if not version_dir.is_dir():
                continue
            version_name = version_dir.name
            xml_path = version_dir / "metadata.xml"

            if not xml_path.exists():
                print(f"[WARNING] Missing metadata.xml in {version_dir}")
                continue

            dbl_id, file_to_passage = parse_metadata(xml_path)
            if not dbl_id:
                continue

            video_files = {f.name: f for f in version_dir.glob("*.mp4")}
            video_entries = []

            for filename, passage in file_to_passage.items():
                if filename in video_files:
                    video_entries.append(
                        {
                            "mp4_path": str(video_files[filename]),
                            "bible_passage": passage,
                        }
                    )
                else:
                    print(f"[WARNING] Missing file: {filename} in {version_dir}")

            if video_entries:
                grouped.append(
                    {
                        "language_code": language_code,
                        "version_name": version_name,
                        "dbl_id": dbl_id,
                        "videos": video_entries,
                    }
                )

    return grouped





def main():
    parser = argparse.ArgumentParser(
        description="Associate sign Bible videos with their scripture passages."
    )
    parser.add_argument(
        "base_dir", type=Path, help="Base directory containing language subfolders"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output path for the resulting JSON file",
        default=Path("video_file_passages_grouped.json"),
    )

    args = parser.parse_args()

    grouped_data = collect_grouped_metadata(args.base_dir)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote grouped data to {args.output_json}")


if __name__ == "__main__":
    main()
# BASE_DIR = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads")
# OUTPUT_JSON = "video_file_passages_grouped.json"


# python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/DBL-sign/collect_video_metadata.py /opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads/