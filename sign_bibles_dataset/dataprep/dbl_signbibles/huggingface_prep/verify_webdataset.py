import argparse
import io
import json
import tarfile
from pathlib import Path

from pose_format import Pose
from tqdm import tqdm


def validate_webdataset(webdataset_dir: Path, max_samples: int | None = None):
    tar_files = sorted(webdataset_dir.glob("**/*.tar"))
    print(f"Found {len(tar_files)} tar files under {webdataset_dir}")

    total_samples = 0
    errors = []

    for tar_path in tqdm(tar_files, desc="Processing shards"):
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                sample_names = set(m.name.split(".")[0] for m in members)

                for sample_name in tqdm(sample_names, desc=f"{tar_path.name} samples", leave=False):
                    total_samples += 1
                    if max_samples and total_samples > max_samples:
                        print("Reached max sample count, exiting early")
                        return

                    try:
                        # Collect all files for this sample
                        files = {m.name.split(".", 1)[1]: m for m in members if m.name.startswith(sample_name)}

                        # Check metadata JSON
                        json_member = files.get("json") or files.get("json_data")
                        assert json_member, f"Missing JSON metadata in {sample_name}"
                        json_bytes = tar.extractfile(json_member).read()
                        metadata = json.loads(json_bytes)

                        # Check video (optional)
                        mp4_member = files.get("mp4")
                        if mp4_member:
                            mp4_bytes = tar.extractfile(mp4_member).read()
                            assert len(mp4_bytes) > 0, f"Empty mp4 file in {sample_name}"

                        # Check pose
                        pose_member = files.get("pose-mediapipe.pose")
                        if pose_member:
                            pose_bytes = tar.extractfile(pose_member).read()
                            pose = Pose.read(io.BytesIO(pose_bytes))
                            assert pose.body.data.shape[0] > 0, f"Pose data empty in {sample_name}"

                        # Optional transcripts check
                        transcripts_member = files.get("transcripts.json")
                        if transcripts_member:
                            transcripts_bytes = tar.extractfile(transcripts_member).read()
                            transcripts = json.loads(transcripts_bytes)
                            assert isinstance(transcripts, list), f"Transcripts not a list in {sample_name}"

                    except Exception as e:
                        errors.append((tar_path.name, sample_name, str(e)))
                        print(f"[ERROR] {tar_path.name} {sample_name}: {e}")

        except Exception as e:
            print(f"[FATAL] Error reading {tar_path}: {e}")
            continue

    print(f"\nTotal samples checked: {total_samples}")
    print(f"Errors found: {len(errors)}")
    for tar_name, sample_name, err in errors:
        print(f"- {tar_name} / {sample_name}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Validate local WebDataset shards before upload.")
    parser.add_argument("webdataset_dir", type=Path, help="Path to local WebDataset directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum samples to check")
    args = parser.parse_args()

    validate_webdataset(args.webdataset_dir, args.max_samples)


if __name__ == "__main__":
    main()