# Sign Bibles Processing

A comprehensive toolkit for processing, segmenting, and preparing sign language Bible videos for machine learning applications.

## Overview

This repository contains a collection of tools designed to work with sign language Bible videos from the Digital Bible Library (DBL). The toolkit provides end-to-end processing capabilities from downloading videos to preparing them for use in machine learning models on platforms like HuggingFace.

## Components

### DBL-sign

A Python-based tool for downloading and managing sign language video content from the Digital Bible Library (DBL).

**Key Features:**
- Modern GUI with progress tracking
- Automatic manifest management
- Optional S3 upload support
- MP4 validation
- Download resumption capability

**Main Scripts:**
- `DBL-manifest-downloader.py`: Downloads sign language videos from DBL
- `DBL-manifest-generator.py`: Generates and manages manifests
- `dbl_utils.py`: Helper utilities for the downloader

### sign-segmentation

A toolkit for precise segmentation and detection of sign language elements in videos, focusing on hands, face, and body pose.

**Key Features:**
- Person detection in video frames
- Hand and face detection using YOLOv8 and pose-based methods
- Pose estimation using DWpose
- Creation of refined segmentation masks
- Combined approach for optimal detection results

**Main Scripts:**
- `refine_mask_v2.py`: Latest version of the mask refinement process
- `video_to_pose_mask.py`: Handles pose estimation and creates ROI masks
- `segment_video.py`: Segments videos for processing

### huggingface-prep

Tools to prepare sign language videos for use with HuggingFace datasets.

**Key Features:**
- Processing videos to extract segments, pose data, and segmentation masks
- Creating WebDataset format with multiple data types
- Including copyright and permission data
- Tracking segment timing information
- Interactive GUI progress display

**Main Scripts:**
- `prepare_webdataset.py`: Main script for preparing data for HuggingFace
- `upload_to_huggingface.py`: Handles uploading processed data to HuggingFace
- `run_with_gui.py`: GUI interface for the preparation process

## Processing Pipeline

The complete processing pipeline follows these steps:

1. **Download**: Fetch sign language Bible videos from DBL using the DBL-sign tools
2. **Segment**: Process videos to extract pose information and segmentation masks using sign-segmentation
3. **Prepare**: Format the processed data into WebDataset format for machine learning using huggingface-prep
4. **Upload**: Optionally upload the prepared datasets to HuggingFace

## Usage

Each component has its own usage instructions in its respective directory. For a complete end-to-end processing workflow:

1. Download videos:
   ```bash
   cd DBL-sign
   python DBL-manifest-downloader.py
   ```

2. Process videos for segmentation:
   ```bash
   cd ../sign-segmentation
   python refine_mask_v2.py "path/to/video.mp4" "output_directory"
   ```

3. Prepare for HuggingFace:
   ```bash
   cd ../huggingface-prep
   python prepare_webdataset.py --num-videos 10 --output-dir ./output
   ```

## Requirements

Each component has its own requirements.txt file with specific dependencies. The main requirements across all components include:

- Python 3.8+
- PyTorch
- OpenCV
- FFmpeg
- Ultralytics YOLOv8
- DWpose
- Various Python packages (see individual requirements.txt files)

## To Do

- [ ] Improve segmentation for better accuracy and performance
- [ ] Test processing pipeline on GPU for faster processing
- [ ] Implement functionality to read videos directly from HuggingFace upload

## License

This code is provided under the same license as the signbibles-processing repository.
