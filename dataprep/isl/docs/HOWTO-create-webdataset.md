# Web Dataset

Our dataset is about a 100 GB in size and each files in it are also large in size like mp4s. Hugging Face provides a library for such usecases, called [webdataset](https://huggingface.co/docs/hub/en/datasets-webdataset). It allows streaming large files.

The webdataset requires the data samples to be made available in the certain format. It includes;
- File naming convention
- Tar archives

## File naming convention
All the files coming under the same data sample should be named with a common base. The remaining part of the filename becomes the feature name to access it.
>  <id>.mp4, <id>.json, <id>.transcript.json, <id>.pose-mediapipe.pose, <id>.pose-dwpose.npz are how we name the files in our dataset.

When using this dataset in code, the user can access these features like shown below

```python
for sample in dataset:
    json_data = sample['json']

    text_json = sample['transcripts.json']

    mp4_bytes = sample['mp4']

    dwpose_coords = sample["pose-dwpose.npz"]

    pose_format_bytes = sample["pose-mediapipe.pose"]
```

## Tar archives and dataset splits

The webdataset also expects the data to be sharded into tar archives of less than 1 GB each. For that we follow this procedure:

1. Group all files based on their base-name first to ensure that files of same sample do not get split accross different tar files.
1. Shuffle and split the list of base-names into train, test and validation sets in the ratio 60:20:20. 
1. Iterate over the list of samples in the splits, estimate their file sizes(of all 5 files in each sample) and chunk them into sets of less the 1 GB.
1. Build tar files for each chunk including all 5 files associated with it. The tar files are to be named in the pattern `shard_<count>-<split>.tar`. Eg: `shard_00005-train.tar`.

## Directory structure, datacard and dataset_stats.json

As this dataset is part of the `BibleNLP/sign-bibles` dataset, the same directory structure is followed:
```
- languagecode
    - projectname
        - tar files
```
The `README.md` added in the huggingface repo will serve as the dataset card. It should include basic details about the dataset, license and code examples for using the dataset. The config section provided in the start of the file will allow hugging face to show dataset preview. It is very helpful for users to see the splits, features and the data itself.

The `BibleNLP/sign-bibles` dataset also includes a `dataset_stats.json` file to note number of shards and samples in each of its subset. In this repo also we follow that pattern.

## Test the Webdataset

Use `test_webdataset.py` to load the tar files from local path or huggingface and test.
