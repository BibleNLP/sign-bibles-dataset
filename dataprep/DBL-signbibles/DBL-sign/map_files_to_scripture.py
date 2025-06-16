import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads")

def parse_metadata(xml_path: Path) -> dict:
    """
    Parse the metadata.xml and return a mapping of filename -> scripture passage.
    """
    file_to_passage = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for pub in root.findall("./publications/publication"):
            for division in pub.findall("./structure/division"):
                passage = division.attrib.get("role", "").strip()
                for content in division.findall("content"):
                    src = content.attrib.get("src", "").strip()
                    filename = Path(src).name
                    file_to_passage[filename] = passage

    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML: {xml_path} - {e}")
    return file_to_passage

def collect_video_mappings(base_dir: Path) -> pd.DataFrame:
    records = []

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
                print(f"[WARNING] Missing metadata.xml in: {version_dir}")
                continue

            file_to_passage = parse_metadata(xml_path)

            video_files = list(version_dir.glob("*.mp4"))
            existing_filenames = {f.name: f for f in video_files}

            for filename, passage in file_to_passage.items():
                if filename in existing_filenames:
                    records.append({
                        "language_code": language_code,
                        "version_name": version_name,
                        "mp4_path": str(existing_filenames[filename]),
                        "bible_passage": passage
                    })
                else:
                    print(f"[WARNING] File listed in metadata but not found: {filename} in {version_dir}")

    return pd.DataFrame.from_records(records)

def main():
    df = collect_video_mappings(BASE_DIR)
    print(df.head())
    df.to_csv("video_file_passages.csv", index=False)

    # count language_code unique values 

    # bible_passage unique values
    for bible_passage in df["bible_passage"].unique():
        print(bible_passage)

if __name__ == "__main__":
    main()
