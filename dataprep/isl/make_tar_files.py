import os
import tarfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
LOG_FILE = "/mnt/share/logs/make_tars.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

def get_file_groups(source_dir):
    """Group files by base name before extension."""
    try:
        files = os.listdir(source_dir)
    except Exception as e:
        logging.error(f"Failed to list directory {source_dir}: {e}")
        raise
    grouped = {}
    for file in files:
        parts = file.split(".")
        if parts[0].isdigit():
            base = parts[0]
            grouped.setdefault(base, []).append(file)
    logging.info(f"Grouped {len(grouped)} file sets from {source_dir}")
    return grouped

def split_grouped_files(group_keys, test_ratio=0.2, val_ratio=0.2, random_state=50):
    # First split: train and temp (temp will be split into val + test)
    logging.debug(f"group_keys: {group_keys}")
    train, temp = train_test_split(group_keys,
                        test_size=test_ratio+val_ratio,
                        random_state=random_state)

    # Second split: validation and test from temp
    val, test = train_test_split(temp,
                        test_size=test_ratio/(test_ratio+val_ratio),
                        random_state=random_state)

    logging.info(f"Split into train: {len(train)}, val: {len(val)}, test: {len(test)}")
    logging.debug(f"Train keys: {train}")
    logging.debug(f"Validation keys: {val}")
    logging.debug(f"Test keys: {test}")
    return { "train": train, "test": test, "val": val}

def group_files_into_chunks(grouped, splits, base_path, max_size):
    """Group the file sets into chunks of total size < max_size."""
    chunks = []
    current_chunk = []
    current_size = 0

    for split, bases in splits.items():
        for base in bases:
            files = grouped[base]
            full_paths = [os.path.join(base_path, f) for f in files]
            try:
                group_size = sum(os.path.getsize(p) for p in full_paths)
            except Exception as e:
                logging.error(f"Failed to get size for files in group {base}: {e}")
                continue

            if current_size + group_size > max_size and current_chunk:
                chunks.append((current_chunk, split))
                current_chunk = []
                current_size = 0

            current_chunk.append((base, full_paths))
            current_size += group_size

        if current_chunk:
            chunks.append((current_chunk, split))
            current_chunk = []
            current_size = 0

    logging.info(f"Grouped into {len(chunks)} chunks (max size {max_size} bytes)")
    logging.debug(f"Chunks detail: {chunks}")
    return chunks

def create_tarballs(chunks, output_dir, count_start):
    for idx, chunk in enumerate(chunks, 1):
        num = count_start+idx
        tar_name = os.path.join(output_dir, f"shard_{num:05d}-{chunk[1]}.tar")
        try:
            with tarfile.open(tar_name, "w") as tar:
                for base, filepaths in chunk[0]:
                    for filepath in filepaths:
                        arcname = os.path.basename(filepath)
                        try:
                            tar.add(filepath, arcname=arcname)
                        except Exception as e:
                            logging.error(f"Failed to add {filepath} to {tar_name}: {e}")
            logging.info(f"Created {tar_name}")
            print(f"Created {tar_name}")
        except Exception as e:
            logging.error(f"Failed to create tarball {tar_name}: {e}")


if __name__ == '__main__':
    SOURCE_DIR = "/mnt/share/ISLGospel_processed"
    OUTPUT_DIR = "/mnt/share/ISLGospel_shards"
    COUNT_START = 0
    MAX_TAR_SIZE = 1 * 1024 * 1024 * 1024 # 1 GB

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        grouped_files = get_file_groups(SOURCE_DIR)
        print(f"Grouped files: {len(grouped_files)}")
        splits = split_grouped_files(sorted(grouped_files.keys()))
        chunks = group_files_into_chunks(grouped_files, splits, SOURCE_DIR, MAX_TAR_SIZE)
        print(f"Created chunks: {len(chunks)}")
        # create_tarballs(chunks, OUTPUT_DIR, count_start=COUNT_START)
        # logging.info("make_tar_files.py completed successfully.")
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise
