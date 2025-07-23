import argparse
import io
import json
from pathlib import Path

import pandas as pd
from datasets import Video, get_dataset_config_names, load_dataset
from pose_format import Pose
from torchvision.io import write_png
from tqdm import tqdm

# requires torchcodec, torchvision


def iterate_over_dataset(language_subset: str, sample_count: int, output_dir: Path | None = None) -> pd.DataFrame:
    """Extract valid queries from the Sign Bibles dataset."""
    # https://huggingface.co/docs/datasets/en/video_load#webdataset
    # The dataet is massive! Best to stream it
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train")

    if output_dir is None:
        output_dir = Path.cwd() / "example_outputs/"
        output_dir.mkdir(exist_ok=True)

    print(f"Loading dataset subset '{language_subset}: {ds}'")
    ds = ds.cast_column("mp4", Video())  # for convenience, otherwise we just get Value('binary'), aka just bytes

    for sample in tqdm(ds.take(sample_count), desc="Parsing samples"):
        sample_key = sample["__key__"]
        # load metadata
        metadata = sample["json"]
        print(json.dumps(metadata, indent=2))
        total_frames = metadata["total_frames"]  # total frame count of the video

        # load Pose format. Normally it expects a file buffer, so..
        pose = Pose.read(io.BytesIO(sample["pose-mediapipe.pose"]))
        print("Pose Format (https://github.com/sign-language-processing/pose)")
        print(pose)  # should show Pose Header

        # DWPose
        if "pose-dwpose.npz" in sample:
            dwpose = sample["pose-dwpose.npz"]
            print("DWPose Format")
            print(type(dwpose))  # dictionary with "frames" as the key, sometimes also "confidences"

        # load video
        video = sample["mp4"]
        print(type(video))  # bytes
        first_frame = video.get_frame_at(0)  # torchcodec Frame type
        print(type(first_frame))
        print(first_frame.data.shape)

        # load transcripts and get start/end frames for a segment
        if "transcripts.json" in sample:
            transcripts = sample["transcripts.json"]
            print(f"Transcripts for {sample_key}")
            for i, transcript in enumerate(transcripts):
                start_frame_index = transcript["start_frame"]
                end_frame_index = transcript["end_frame"]

                mid_index = start_frame_index + (end_frame_index - start_frame_index) // 2

                print(transcript["text"][:100], start_frame_index, end_frame_index)

                # torchcodec._frame.Frame's .data has the data
                # start_frame = video.get_frame_at(start_frame_index).data
                mid_frame = video.get_frame_at(mid_index).data
                # end_frame = video.get_frame_at(end_frame_index).data

                # start_out_path = output_dir / f"{sample_key}_{start_frame_index}.png"
                # print(f"Writing to {Path(start_out_path).resolve()}")
                # write_png(start_frame, start_out_path)

                # end_out_path = output_dir / f"{sample_key}_{end_frame_index}.png"
                # print(f"Writing to {Path(end_out_path).resolve()}")
                # write_png(end_frame, end_out_path)

                mid_out_path = output_dir / f"{sample_key}_seg{i}_frame{mid_index}.png"
                print(f"Writing to {Path(mid_out_path).resolve()}")
                write_png(mid_frame, mid_out_path)


def main(
    language_subset: str,
    sample_count: int,
):
    iterate_over_dataset(language_subset=language_subset, sample_count=sample_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Sign Bibles queries from Huggingface dataset.")
    parser.add_argument(
        "--language-subset",
        type=str,
        default="ase_small",
        help="Language subset to download (default: ase_small)",
    )
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default="./example_outputs/")

    args = parser.parse_args()

    if args.language_subset is not None:
        configs = [args.language_subset]
    else:
        configs = get_dataset_config_names("bible-nlp/sign-bibles")
        print(f"Available configs: {configs}")

    for language_subset in configs:
        main(language_subset=language_subset, sample_count=args.sample_count)
