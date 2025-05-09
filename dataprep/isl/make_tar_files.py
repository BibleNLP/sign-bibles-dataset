import os
import tarfile
from pathlib import Path

# CONFIGURATION
# SOURCE_DIR = "/path/to/your/files"
# OUTPUT_DIR = "/path/to/output/tars"
# MAX_TAR_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB


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

def group_files_into_chunks(grouped, base_path, max_size):
    """Group the file sets into chunks of total size < max_size."""
    chunks = []
    current_chunk = []
    current_size = 0

    for base, files in sorted(grouped.items(), key=lambda x: int(x[0])):
        full_paths = [os.path.join(base_path, f) for f in files]
        group_size = sum(os.path.getsize(p) for p in full_paths)

        if current_size + group_size > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append((base, full_paths))
        current_size += group_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def create_tarballs(chunks, output_dir, count_start):
    for idx, chunk in enumerate(chunks, 1):
        num = count_start+idx
        tar_name = os.path.join(output_dir, f"chunk_{num:05d}.tar")
        with tarfile.open(tar_name, "w") as tar:
            for base, filepaths in chunk:
                for filepath in filepaths:
                    arcname = os.path.basename(filepath)
                    tar.add(filepath, arcname=arcname)
        print(f"Created {tar_name}")

if __name__ == '__main__':
    # Main Execution
    # SOURCE_DIR = "/mnt/nextcloud/ISLGospels_processed"
    # OUTPUT_DIR = "/mnt/nextcloud/ISL_dataset_chunks0"

    SOURCE_DIR = "../../../ISLGospels_processed"
    OUTPUT_DIR = "../../../ISLGospels_tar_chunks"
    COUNT_START = 14
    MAX_TAR_SIZE = 1 * 1024 * 1024 * 1024 # 1 GB

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grouped_files = get_file_groups(SOURCE_DIR)
    print(f"Grouped files: {len(grouped_files)}")
    chunks = group_files_into_chunks(grouped_files, SOURCE_DIR, MAX_TAR_SIZE)
    print(f"Created chunks: {len(chunks)}")
    create_tarballs(chunks, OUTPUT_DIR, count_start=COUNT_START)
