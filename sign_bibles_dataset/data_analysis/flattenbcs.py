import logging
import shutil
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def flatten_subfolders(base_dir: Path) -> None:
    """
    Flatten subfolders by moving files up one level with subfolder name prepended.

    Example:
        shard_00001-train/8.json -> shard_00001-train_8.json

    """
    if not base_dir.exists() or not base_dir.is_dir():
        logging.error("Base directory does not exist or is not a directory: %s", base_dir)
        return

    subfolders = [p for p in base_dir.iterdir() if p.is_dir()]
    logging.info("Found %d subfolders to process.", len(subfolders))

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        for file in subfolder.iterdir():
            if file.is_file():
                new_name = f"{subfolder.name}_{file.name}"
                new_path = base_dir / new_name
                if new_path.exists():
                    logging.warning("File already exists, skipping: %s", new_path)
                    continue
                shutil.move(str(file), str(new_path))
        # After moving, remove the empty subfolder
        try:
            subfolder.rmdir()
        except OSError:
            logging.warning("Subfolder not empty, skipping removal: %s", subfolder)


if __name__ == "__main__":
    base_dir = Path(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/nas_data/DBL_Deaf_Bibles/webdataset_extracted/ins/BridgeConnISLV_in_Indian_Delhi_Sign_Language"
    )
    flatten_subfolders(base_dir=base_dir)
