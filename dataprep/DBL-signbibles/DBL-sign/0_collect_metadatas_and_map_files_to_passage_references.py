import xml.etree.ElementTree as ET
from pathlib import Path
import json

BASE_DIR = Path(
    "/opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/DBL_Deaf_Bibles/sign-bibles-dataset-script-downloads"
)
# BASE_DIR = Path(
#     r"/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/downloads/esl/"
# )
OUTPUT_JSON = "video_file_passages.json"


def parse_metadata(xml_path: Path):
    """
    Parse a metadata.xml and return:
    - DBL id
    - mapping of video filename -> passage
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        dbl_id = root.attrib.get("id", None)
        file_to_passage = {}

        for pub in root.findall("./publications/publication"):
            for division in pub.findall("./structure/division"):
                passage = division.attrib.get("role", "").strip()
                for content in division.findall("content"):
                    src = content.attrib.get("src", "").strip()
                    filename = Path(src).name
                    file_to_passage[filename] = passage

        return dbl_id, file_to_passage

    except ET.ParseError as e:
        print(f"[ERROR] Could not parse {xml_path}: {e}")
        return None, {}


def collect_video_metadata(base_dir: Path):
    results = []

    for language_dir in base_dir.iterdir():
        if not language_dir.name == "esl":
            continue
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

            dbl_id, file_to_passage = parse_metadata(xml_path)
            if not dbl_id:
                print(f"[WARNING] No DBLMetadata ID found in: {xml_path}")
                continue

            video_files = {f.name: f for f in version_dir.glob("*.mp4")}

            for filename, passage in file_to_passage.items():
                if filename in video_files:
                    file_metadata =  {
                            "language_code": language_code,
                            "project_name": version_name,
                            # "dbl_id": dbl_id,
                            "source": f"https://app.thedigitalbiblelibrary.org/entry?id={dbl_id}",
                            "filename": filename,
                            "mp4_path": str(video_files[filename]),
                            "bible-ref": passage,
                        }
                    out_path = Path(video_files[filename]).with_suffix(".json")
                    
                    # print(out_path)
                    
                    results.append(
                        file_metadata
                    )
                    with open(out_path, "w") as out_f:
                        del file_metadata["mp4_path"]
                        json.dump(file_metadata, out_f)
                    # exit()

                else:
                    print(
                        f"[WARNING] File listed in metadata but not found: {filename} in {version_dir}"
                    )

    return results


def main():
    all_metadata = collect_video_metadata(BASE_DIR)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote {len(all_metadata)} video entries to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
