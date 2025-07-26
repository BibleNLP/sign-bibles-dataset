"""Script to run deepface and try to find unique signers and race/gender demographics. Does not work"""
import argparse
import logging
from pathlib import Path

import pandas as pd
from deepface import DeepFace
from tqdm import tqdm


def find_and_analyze_faces(input_dir: Path):
    png_paths = list(input_dir.rglob("frame_*.png"))
    if not png_paths:
        logging.warning("No matching images found.")
        return

    db_path = input_dir
    db_path.mkdir(exist_ok=True)

    found_faces = set()

    for img_path in tqdm(png_paths, desc="Processing frames", disable=True):
        try:
            # Attempt to find the face in DB
            results = DeepFace.find(
                img_path=str(img_path),
                db_path=str(db_path),
                enforce_detection=False,
                # detector_backend="yolov8",
                # detector_backend="dlib",
                # model_name="ArcFace",
                # model_name="Dlib",
                # threshold=99,
            )
            # print(results)

            if results and isinstance(results, list) and isinstance(results[0], pd.DataFrame):
                matches_df = results[0]
                if not matches_df.empty:
                    face_id = matches_df.iloc[0]["identity"]
                    found_faces.add(face_id)

        except (FileNotFoundError, OSError) as e:
            logging.warning(f"I/O error on {img_path}: {e}")
        except ValueError as e:
            logging.warning(f"Value error on {img_path}: {e}")

    logging.info(f"Unique faces identified: {len(found_faces)}")


def main():
    parser = argparse.ArgumentParser(description="Count unique faces in directory using DeepFace.")
    parser.add_argument("input_dir", type=str, help="Directory containing frame_*.png images")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s"
    )

    input_path = Path(args.input_dir)
    if not input_path.exists():
        logging.error(f"Directory not found: {input_path}")
        return

    find_and_analyze_faces(input_path)


if __name__ == "__main__":
    main()
