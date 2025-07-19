from pathlib import Path
import argparse

import pandas as pd
from datasets import Video, get_dataset_config_names, load_dataset
from tqdm import tqdm
from PIL import Image
from torchvision.io import write_png
# requires torchcodec, torchvision


def iterate_over_dataset(language_subset: str, sample_count: int) -> pd.DataFrame:
    """Extract valid queries from the Sign Bibles dataset."""
    # https://huggingface.co/docs/datasets/en/video_load#webdataset
    # The dataet is massive! Best to stream it
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train")

    print(f"Loading dataset subset '{language_subset}: {ds}'")
    ds = ds.cast_column("mp4", Video())  # for convenience, otherwise we just get Value('binary'), aka just bytes

    for sample in tqdm(ds.take(sample_count), desc="Parsing samples"):
        sample_key = sample["__key__"]
        # load metadata
        metadata = sample["json"]
        total_frames = metadata["total_frames"]

        # load video
        video = sample["mp4"]
        print(type(video))  # bytes
        first_frame = video.get_frame_at(0)
        print(type(first_frame))

        print(first_frame.data.shape)

        # load transcripts
        if "transcripts.json" in sample:
            transcripts = sample["transcripts.json"]
            print(f"Transcripts for {sample_key}")
            for transcript in transcripts:
                start_frame_index = transcript["start_frame"]

                end_frame_index = transcript["end_frame"]
                print(transcript["text"][:100], start_frame_index, end_frame_index)

                # torchcodec._frame.Frame's .data has the data
                start_frame = video.get_frame_at(start_frame_index).data
                end_frame = video.get_frame_at(end_frame_index).data

                start_out_path = f"{sample_key}_{start_frame_index}.png"
                print(f"Writing to {Path(start_out_path).resolve()}")
                write_png(start_frame, start_out_path)

                end_out_path = f"{sample_key}_{end_frame_index}.png"
                print(f"Writing to {Path(end_out_path).resolve()}")
                write_png(end_frame, end_out_path)


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
    parser.add_argument("--sample-count", type=int, default=1)

    args = parser.parse_args()

    if args.language_subset is not None:
        configs = [args.language_subset]
    else:
        configs = get_dataset_config_names("bible-nlp/sign-bibles")
        print(f"Available configs: {configs}")

    for language_subset in configs:
        main(language_subset=language_subset, sample_count=args.sample_count)
