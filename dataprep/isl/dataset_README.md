---
configs:
  - config_name: all
    data_files:
      - split: train
        path: "*/*/*-train.tar"
      - split: val
        path: "*/*/*-val.tar"
      - split: test
        path: "*/*/*-test.tar"
    default: true
  - config_name: indian_sign_language_version_islv
    data_files:
      - split: train
        path: "ins/indian_sign_language_version_islv/*-train.tar"
      - split: val
        path: "ins/indian_sign_language_version_islv/*-val.tar"
      - split: test
        path: "ins/indian_sign_language_version_islv/*-test.tar"
language:
- ins
license: cc-by-sa-4.0
datasets:
- bible-nlp/sign-bibles
- bridgeconn/sign-bibles-isl
tags:
- video
- bible
- translation
- multilingual
- religious-text
- parallel-corpus
- low-resource-languages
---


# Dataset Card for Sign Bible Dataset

This dataset contains Indian sign language videos from the Bible recordings, processed for machine learning applications.
The dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).


## Dataset Details

Contains 1114 sign videos of 19.5 hours in total.

### Dataset Description

- Continuous signing videos
- Parallel text
- Bible reference to signed verses
- Pose estimation data in the following formats
  - Frames wise body landmarks detected by dwpose as a numpy array
  - Frames wise body landmarks detected by mediapose as .pose format
- Signer id for each video

## Uses

- Sign video and corresponding pose estimation data for pose based applications
- Parallel Sign language - text for translation purposes


## How to use

```python
import webdataset as wds
from torch.utils.data import DataLoader
import numpy as np
import json
import tempfile
import os
import cv2


def main():
    lang = "ins" 
    project = "indian_sign_language_version_islv"
    split = "train"
    buffer_size = 1024
    dataset = (
        wds.WebDataset(
        f"https://huggingface.co/datasets/bridgeconn/sign-bibles-isl/resolve/main/{lang}/{project}/shard_{{00001..00002}}-{split}.tar",
        shardshuffle=False)
        .shuffle(buffer_size)
        .decode()
    )
    for sample in dataset:
        ''' Each sample contains:
             'mp4', 'pose-dwpose.npz', 'pose-mediapipe.pose'
             'transcripts.json' and 'json'
        '''
        # print(sample.keys())

        # JSON metadata
        json_data = sample['json']
        print(json_data['total_frames'])
        print(json_data['bible-ref'])
        print(json_data['biblenlp-vref'])
        print(json_data['signer'])

        # Text
        text_json = sample['transcripts.json']
        print(text_json[0]['text'])
        print(text_json[0]['language']['name'])

        # main video
        mp4_data = sample['mp4']
        process_video(mp4_data)

        # dwpose results
        dwpose_coords = sample["pose-dwpose.npz"]
        frame_poses = dwpose_coords['frames'].tolist()
        print(f"Frames in dwpose coords: {len(frame_poses)} poses")
        print(f"Pose coords shape: {len(frame_poses[0][0])}")
        print(f"One point looks like [x,y]: {frame_poses[0][0][0]}")

        # mediapipe results in .pose format
        pose_format_data = sample["pose-mediapipe.pose"]
        process_poseformat(pose_format_data)

        break

def process_poseformat(pose_format_data):
	from pose_format import Pose
	temp_file = None
	try:
		with tempfile.NamedTemporaryFile(suffix=".pose", delete=False) as tmp:
			tmp.write(pose_format_data)
			temp_file = tmp.name

		data_buffer = open(temp_file, "rb").read()
		pose = Pose.read(data_buffer)

		print(f"Mediapipe results from pose-format: {pose.body.data.shape}")
	except Exception as e:
		print(f"Error processing pose-format: {e}")
	finally:
		if temp_file and os.path.exists(temp_file):
			os.remove(temp_file) # Clean up the temporary file


def process_video(mp4_data):
	print(f"Video bytes length: {len(mp4_data)} bytes")

	temp_file = None
	try:
		# Processing video from temporary file
		with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
			tmp.write(mp4_data)
			temp_file = tmp.name

		cap = cv2.VideoCapture(temp_file)

		if not cap.isOpened():
			raise IOError(f"Could not open video file: {temp_file}")

		# Example: Get video metadata
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = cap.get(cv2.CAP_PROP_FPS)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		print(f"Video Info: {frame_count} frames, {fps:.2f} FPS, {width}x{height}")

		# Example: Read and display the first frame (or process as needed)
		ret, frame = cap.read()
		if ret:
			print(f"First frame shape: {frame.shape}, dtype: {frame.dtype}")
			# You can then use this frame for further processing, e.g.,
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			import matplotlib.pyplot as plt
			plt.imshow(frame_rgb)
			plt.title(f"Sample First Frame")
			plt.show()
		else:
			print("Could not read first frame.")

		cap.release()

	except Exception as e:
		print(f"Error processing external MP4: {e}")
	finally:
		if temp_file and os.path.exists(temp_file):
			os.remove(temp_file) # Clean up the temporary file


if __name__ == '__main__':
	main()
```
