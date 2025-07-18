# TODO: add tasks like in https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/blob/main/README.md or https://huggingface.co/datasets/racineai/OGC_MEGA_MultiDomain_DocRetrieval/blob/main/README.md or https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/README.md, maybe also text-to-video like https://huggingface.co/datasets/nkp37/OpenVid-1M
def get_usage() -> str:
    usage_str = """## Usage (In Progress, untested)

```python
from datasets import load_dataset
import json

# Load dataset from Hugging Face Hub
dataset = load_dataset("bible-nlp/sign-bibles", subset="ase")

# Iterate through samples
for sample in dataset["train"]:
    # Access components
    original_video = sample["filename"]

    # Corresponding BSB version Bible verses, if available
    transcripts = sample["transcripts"]

    # pose-format file (https://github.com/sign-language-processing/pose)
    pose = sample["pose"]["mediapipe"]

    # DWPose file
    pose = sample["pose"]["DWPose"]
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
