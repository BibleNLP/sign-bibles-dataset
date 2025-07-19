# TODO: add tasks like in https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/main/README.md or https://huggingface.co/datasets/racineai/OGC_MEGA_MultiDomain_DocRetrieval/blob/main/README.md or https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/README.md, maybe also text-to-video like https://huggingface.co/datasets/nkp37/OpenVid-1M
def get_usage() -> str:
    usage_str = """## Usage (In Progress, untested)

```python

## Usage (Tested on Linux)

Requires installing torchcodec, torchvision

```python
import argparse
import io
import json
from pathlib import Path

import pandas as pd
from datasets import Video, get_dataset_config_names, load_dataset
from pose_format import Pose
from torchvision.io import write_png
from tqdm import tqdm


def iterate_over_dataset(language_subset: str, sample_count: int) -> pd.DataFrame:
    \"\"\"Extract valid queries from the Sign Bibles dataset.\"\"\"
    # https://huggingface.co/docs/datasets/en/video_load#webdataset
    # The dataset is massive! Best to stream it
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train")

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
        print(pose)

        # DWPose
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
            for transcript in transcripts:
                start_frame_index = transcript["start_frame"]

                end_frame_index = transcript["end_frame"]
                print(transcript["text"][:100], start_frame_index, end_frame_index)

                # torchcodec._frame.Frame's .data has the data
                start_frame = video.get_frame_at(start_frame_index).data
                end_frame = video.get_frame_at(end_frame_index).data

                start_out_path = f"{sample_key}_{start_frame_index}.png"
                print(f"Writing to {Path(start_out_path).resolve()}")
                # write_png(start_frame, start_out_path)

                end_out_path = f"{sample_key}_{end_frame_index}.png"
                print(f"Writing to {Path(end_out_path).resolve()}")
                # write_png(end_frame, end_out_path)


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
```"""
    return usage_str


def get_configs_block(configs: list[tuple[str, str]]) -> str:
    """Generate a YAML block listing configurations per language."""
    lines = ["configs:", "  - config_name: all", "    data_files: '*/*/*.tar'", "    default: true"]
    print(configs)
    for config_name, pattern in list(configs):
        lines.append(f"  - config_name: {config_name.replace('/', '_')}\n    data_files: {pattern}")

    return "\n".join(lines)


def get_tags_block():
    tags_block = """tags:
- video
- bible
- translation
- multilingual
- religious-text
- parallel-corpus
- low-resource-languages"""

    return tags_block


def build_dataset_card_markdown_and_yaml(
    dataset_name: str,
    shard_count: int,
    total_sample_count: int,
    languages: set[str],
    projects: set[str],
    configs: set[tuple[str, str]],
) -> str:
    """Build the dataset card string from metadata."""
    languages_block = "\n".join(f"- {lang}" for lang in sorted(languages))
    projects_list = ", ".join(sorted(projects))

    card = f"""---
{get_configs_block(configs=configs)}
language:
{languages_block}
license: cc-by-sa-4.0
datasets:
- {dataset_name}
{get_tags_block()}
---

# {dataset_name}

**This dataset is still being generated and currently includes only test files**

This dataset contains sign language videos from the Digital Bible Library (DBL), processed for machine learning applications. The dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).

## Dataset Description

- **Size**: {shard_count} shards, approximately {total_sample_count} samples
- **Languages**: {", ".join(sorted(languages))}
- **Projects**: {projects_list}
- **License**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Dataset Structure

Each sample contains:
* ["mp4"] the original video
* ["json"] Metadata, including bible reference, copyright information, language information.
* ["transcripts.json"]: Bible verses from the eBible corpus, as well as [vref indices](https://github.com/BibleNLP/ebible/blob/main/metadata/vref.txt). For some videos, more fine-grained annotations are available with start/end frames for certain sections
* ["pose-mediapipe.pose"]: Mediapipe holistic keypoints, created with [Pose Format](https://github.com/sign-language-processing/pose)
* ["pose-dwpose.npz"]: [DWPose outputs](https://github.com/IDEA-Research/DWPose/tree/onnx/ControlNet-v1-1-nightly/annotator/dwpose), saved as compressed npz files.


{get_usage()}

## License and Attribution

This dataset is derived from the Digital Bible Library (DBL) sign language content.
Each sample includes copyright and license information in its metadata.

The dataset is provided under a CC BY-SA 4.0 license, which allows for:
- Sharing: Copy and redistribute the material in any medium or format
- Adaptation: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- ShareAlike: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original

Please refer to the individual sample metadata for specific attribution requirements.
"""
    return card
