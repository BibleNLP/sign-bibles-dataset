#!/usr/bin/env python

import argparse
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from pyarrow.lib import ArrowCapacityError, ArrowInvalid
from tqdm import tqdm


def iterate_keys(language_subset: str, path: Path | None) -> None:
    """Iterate over all samples in the Sign Bibles dataset and print idx and sample key."""

    # https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/loading_methods#datasets.load_dataset
    if path is None:
        print(f"\n==> Loaded dataset config '{language_subset}'")
        ds = load_dataset(
            "bible-nlp/sign-bibles",
            name=language_subset,
            streaming=True,
            split="train",
        )
    else:
        print(f"==> LOADING PATH {path}")
        ds = load_dataset(
            path="webdataset",
            data_files={"train": f"{path}/**/*.tar"},
            streaming=True,
            split="train",
        )

    idx = 0
    estimated_shard_index = 0
    samples_per_shard = 5
    with tqdm(desc=f"{language_subset} samples") as pbar:
        iterator = iter(ds)
        while True:
            try:
                if (
                    idx % samples_per_shard == 0 and idx > 0
                ):  # 5 samples per shard: 0, 1, 2, 3, 4
                    print(
                        f"Estimated Shard idx (starting at 0, {samples_per_shard}/shard): {estimated_shard_index}"
                    )
                    estimated_shard_index += 1
                sample = next(iterator)
                sample_key = sample.get("__key__", "missing-key")
                print(f"[{language_subset}] idx={idx}, key={sample_key}")
                idx += 1

                pbar.update(1)

            except StopIteration:
                print(f"Finished iterating through {idx} samples of {language_subset}")
                break

            except (ArrowCapacityError, ArrowInvalid) as e:
                print(f"PyArrow error on idx={idx}, config={language_subset}: {e}")
                idx += 1
                pbar.update(1)
                continue

            except KeyError as e:
                print(f"Missing key error on idx={idx}, config={language_subset}: {e}")
                idx += 1
                pbar.update(1)
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Iterate through Sign Bibles dataset and print sample keys."
    )
    parser.add_argument("--path", type=Path)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    path = args.path

    if args.config is None:
        configs = get_dataset_config_names("bible-nlp/sign-bibles")
        print(f"Available configs: {configs}")
        # # configs = ["ase_small"]
        # # configs = [c for c in configs if "small" in c and "ase" in c]
        # # configs.append("all")
        # configs = [
        #     "ase_small",
        #     # "ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_expanded_with_more_information_small"
        # ]
    else:
        configs = [args.config]

    for language_subset in configs:
        print(f"TESTING CONFIG {language_subset}, path: {path}")
        iterate_keys(language_subset, path)
        # try:

        # except (ArrowCapacityError, ArrowInvalid) as e:
        #     print(f"PyArrow error at config level for {language_subset}: {e}")
        #     continue
        # except RuntimeError as e:
        #     print(f"RuntimeError at config level for {language_subset}: {e}")
        #     continue


if __name__ == "__main__":
    main()
