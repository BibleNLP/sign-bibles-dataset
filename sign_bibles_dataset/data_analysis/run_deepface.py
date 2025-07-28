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

    # found_faces = set()

    # Lists to accumulate stats
    age_list = []
    gender_list = []
    race_list = []

    for img_path in tqdm(png_paths, desc="Processing frames"):
        try:
            # finds 45 faces in 45 frames
            # results = DeepFace.find(
            #     img_path=str(img_path),
            #     db_path=str(db_path),
            #     enforce_detection=False,
            # )

            # if results and isinstance(results, list) and isinstance(results[0], pd.DataFrame):
            #     matches_df = results[0]
            #     if not matches_df.empty:
            #         face_id = matches_df.iloc[0]["identity"]
            #         found_faces.add(face_id)

            objs = DeepFace.analyze(img_path=str(img_path), actions=["age", "gender", "race"], enforce_detection=True)

            if not isinstance(objs, list):
                objs = [objs]

            print(objs)
            # exit()

            # {'age': 33, 'region': {'x': 1314, 'y': 187, 'w': 431, 'h': 431, 'left_eye': None, 'right_eye': (604, 231)}, 'face_confidence': np.float64(0.91),
            # 'gender': {'Woman': np.float32(41.781376, 'Man': np.float32(79.40061)}, 'dominant_gender': 'Man',
            # 'race': {'asian': np.float32(1.4.543095), 'indian': np.float32(23.83602), 'black': np.float32(7.3527412), 'white': np.float32(17.124534758), 'indian': np.float32(0.5223312), 'black': np.float32(0.0711698), 'white': np.floa 'dominant_race': 'latino hispanic'}
            # t32(83.0013), 'middle eastern': np.float32(6.0256443), 'latino hispanic': np.float32(8.926ye': None}, 'face_confidence': np.float64(0.97), 'gender': {'Woman': np.float32(20.599384), 'Man': n083)}, 'dominant_race': 'white'}

            for obj in objs:
                print(obj)
                if "age" in obj and isinstance(obj["age"], (int, float)):
                    age_list.append(obj["age"])
                if "dominant_gender" in obj:
                    gender_list.append(obj["dominant_gender"])
                if "dominant_race" in obj:
                    race_list.append(obj["dominant_race"])
            # exit()

        except (FileNotFoundError, OSError) as e:
            logging.warning(f"I/O error on {img_path}: {e}")
        except ValueError as e:
            logging.warning(f"Value error on {img_path}: {e}")

    # Create DataFrames
    age_df = pd.DataFrame(age_list, columns=["age"])
    gender_df = pd.DataFrame(gender_list, columns=["gender"])
    race_df = pd.DataFrame(race_list, columns=["race"])

    # Save to parquet
    age_df.to_parquet(input_dir / "age_stats.parquet", index=False)
    gender_df.to_parquet(input_dir / "gender_stats.parquet", index=False)
    race_df.to_parquet(input_dir / "race_stats.parquet", index=False)

    logging.info(f"Saved age, gender, and race stats to: {input_dir}")
    # logging.info(f"Unique faces identified: {len(found_faces)}")


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
