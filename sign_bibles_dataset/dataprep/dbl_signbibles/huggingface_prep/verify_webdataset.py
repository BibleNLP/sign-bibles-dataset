import argparse
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from pose_format import Pose
from sign_bibles_dataset.dataprep.dbl_signbibles.sign_segmentation.recursively_run_segmentation import (
    MODEL_CHOICES,
)
from tqdm import tqdm


def get_expected_extensions() -> set[str]:
    """Return the full set of expected file extensions for each sample."""
    base_extensions = {
        "json",  # base metadata
        "mp4",  # the actual file
        "pose-mediapipe.pose",
        # "pose-dwpose.npz",
        "transcripts.json",
    }

    model_extensions = {
        f"{Path(model).stem.lower()}.eaf" for model in MODEL_CHOICES
    } | {
        f"{Path(model).stem.lower()}.autosegmented_segments.json"
        for model in MODEL_CHOICES
    }

    return base_extensions | model_extensions


def extract_sample_groups(members):
    """Group tar members by shared prefix (everything before first dot)."""
    grouped = defaultdict(list)
    for m in members:
        if "." not in m.name:
            continue  # skip invalid or non-sample files
        prefix, ext = m.name.split(".", 1)
        grouped[prefix].append((ext, m))
    return grouped


def validate_sample(
    sample_name: str, filelist: list[tuple[str, tarfile.TarInfo]], tar, errors
):
    try:
        # Enforce lowercase extension naming
        for ext, _ in filelist:
            if ext != ext.lower():
                raise ValueError(f"{ext}: extension must be lowercased")

        files = {ext: m for ext, m in filelist}

        # Check required files
        required_keys = get_expected_extensions()
        missing_keys = required_keys - set(files.keys())
        if missing_keys:
            raise ValueError(f"Missing required files: {sorted(missing_keys)}")

        # Load and check JSON metadata
        metadata = json.loads(tar.extractfile(files["json"]).read())

        # Validate MP4 content
        mp4_bytes = tar.extractfile(files["mp4"]).read()
        if not mp4_bytes:
            raise ValueError("Empty mp4 file")

        # Validate pose
        pose_bytes = tar.extractfile(files["pose-mediapipe.pose"]).read()
        pose = Pose.read(io.BytesIO(pose_bytes))
        if pose.body.data.shape[0] == 0:
            raise ValueError("Pose data is empty")

        # Validate transcripts
        transcripts = json.loads(tar.extractfile(files["transcripts.json"]).read())
        if not isinstance(transcripts, list):
            raise ValueError("Transcripts should be a list")

    except Exception as e:
        errors.append((sample_name, str(e)))
        print(f"[ERROR] {sample_name}: {e}")


def validate_webdataset(webdataset_dir: Path, max_samples: int | None = None):
    if webdataset_dir.is_file():
        tar_files = [webdataset_dir]
    else:
        tar_files = sorted(webdataset_dir.rglob("*.tar"))
    print(f"Found {len(tar_files)} tar files under {webdataset_dir}")

    total_samples = 0
    total_errors = []
    all_extensions: set[str] = set()

    for tar_path in tqdm(tar_files, desc="Verifying shards"):
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                sample_groups = extract_sample_groups(members)

                for sample_name, filelist in tqdm(
                    sample_groups.items(), desc=tar_path.name, leave=False
                ):
                    total_samples += 1
                    if max_samples and total_samples > max_samples:
                        print("Reached max sample count, exiting early")
                        return

                    # Track all extensions seen
                    for ext, _ in filelist:
                        all_extensions.add(ext)

                    validate_sample(sample_name, filelist, tar, total_errors)

        except Exception as e:
            print(f"[FATAL] Error reading {tar_path}: {e}")
            continue

    print(f"\nTotal samples checked: {total_samples}")
    print(f"Errors found: {len(total_errors)}")
    for sample_name, err in total_errors:
        print(f"- {sample_name}: {err}")

    # Extension audit
    print(f"\nAll extensions found ({len(all_extensions)}):")
    for ext in sorted(all_extensions):
        print(f" - {ext}")

    # Unexpected extensions
    expected_extensions = get_expected_extensions()
    unexpected = all_extensions - expected_extensions
    if unexpected:
        raise ValueError(f"Unexpected extensions found: {sorted(unexpected)}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate local WebDataset shards before upload."
    )
    parser.add_argument(
        "webdataset_dir", type=Path, help="Path to local WebDataset directory"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum samples to check",
    )
    args = parser.parse_args()

    validate_webdataset(args.webdataset_dir, args.max_samples)


if __name__ == "__main__":
    main()
