#!/usr/bin/env python

import argparse
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from pyarrow.lib import ArrowCapacityError, ArrowInvalid


def iterate_keys(language_subset: str) -> None:
    """Iterate over all samples in the Sign Bibles dataset and print idx and sample key."""
    ds = load_dataset("bible-nlp/sign-bibles", language_subset, streaming=True, split="train")
    print(f"\n==> Loaded dataset config '{language_subset}'")

    for idx, sample in enumerate(tqdm(ds, desc=f"{language_subset} samples")):
        try:
            sample_key = sample.get("__key__", "missing-key")
            print(f"[{language_subset}] idx={idx}, key={sample_key}")
        except (ArrowCapacityError, ArrowInvalid) as e:
            print(f"PyArrow error on idx={idx}, config={language_subset}: {e}")
            continue
        except KeyError as e:
            print(f"Missing key error on idx={idx}, config={language_subset}: {e}")
            continue


def main():
    configs = get_dataset_config_names("bible-nlp/sign-bibles")
    print(f"Available configs: {configs}")

    for language_subset in configs:
        print(f"TESTING CONFIG {language_subset}")
        try:
            iterate_keys(language_subset)
        except (ArrowCapacityError, ArrowInvalid) as e:
            print(f"PyArrow error at config level for {language_subset}: {e}")
            continue
        except RuntimeError as e:
            print(f"RuntimeError at config level for {language_subset}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate through Sign Bibles dataset and print sample keys.")
    args = parser.parse_args()

    main()
