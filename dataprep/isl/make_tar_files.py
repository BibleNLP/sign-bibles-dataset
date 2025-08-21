import os
import tarfile
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_file_groups(source_dir):
    """Group files by base name before extension."""
    files = os.listdir(source_dir)
    grouped = {}
    for file in files:
        parts = file.split(".")
        if parts[0].isdigit():
            base = parts[0]
            grouped.setdefault(base, []).append(file)
    return grouped

def split_grouped_files(group_keys, test_ratio=0.2, val_ratio=0.2, random_state=50):
    # First split: train and temp (temp will be split into val + test)
    train, temp = train_test_split(group_keys,
                        test_size=test_ratio+val_ratio,
                        random_state=random_state)

    # Second split: validation and test from temp
    val, test = train_test_split(temp,
                        test_size=test_ratio/(test_ratio+val_ratio),
                        random_state=random_state)

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
            group_size = sum(os.path.getsize(p) for p in full_paths)

            if current_size + group_size > max_size and current_chunk:
                chunks.append((current_chunk, split))
                current_chunk = []
                current_size = 0

            current_chunk.append((base, full_paths))
            current_size += group_size

        if current_chunk:
            chunks.append((current_chunk, split))

    return chunks

def create_tarballs(chunks, output_dir, count_start):
    for idx, chunk in enumerate(chunks, 1):
        num = count_start+idx
        tar_name = os.path.join(output_dir, f"shard_{num:05d}-{chunk[1]}.tar")
        with tarfile.open(tar_name, "w") as tar:
            for base, filepaths in chunk[0]:
                for filepath in filepaths:
                    arcname = os.path.basename(filepath)
                    tar.add(filepath, arcname=arcname)
        print(f"Created {tar_name}")

if __name__ == '__main__':

    SOURCE_DIR = "/mnt/share/ISLGospel_processed"
    OUTPUT_DIR = "/mnt/share/ISLGospel_shards"
    COUNT_START = 0
    MAX_TAR_SIZE = 1 * 1024 * 1024 * 1024 # 1 GB

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grouped_files = get_file_groups(SOURCE_DIR)
    print(f"Grouped files: {len(grouped_files)}")
    splits = split_grouped_files(sorted(grouped_files.keys()))
    chunks = group_files_into_chunks(grouped_files, splits, SOURCE_DIR, MAX_TAR_SIZE)
    print(f"Created chunks: {len(chunks)}")
    create_tarballs(chunks, OUTPUT_DIR, count_start=COUNT_START)
