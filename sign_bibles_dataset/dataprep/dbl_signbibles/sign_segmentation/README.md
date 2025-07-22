# Sign Language Segmentation

A toolkit for precise segmentation and detection of sign language elements in videos, focusing on hands, face, and body pose.

## Overview

This project provides tools to process sign language videos and extract segmentation masks for hands, face, and body poses. It combines multiple detection approaches to create accurate masks that can be used for further analysis or processing of sign language content.

## Features

- **Person Detection**: Identifies people in video frames
- **Hand and Face Detection**: Detects hands and faces using both YOLOv8 and pose-based methods
- **Pose Estimation**: Uses DWpose to extract skeletal keypoints
- **Segmentation Masks**: Creates refined masks for hands and face regions
- **Combined Approach**: Intelligently combines segmentation and pose-based detection for optimal results

## Key Components

### Main Scripts

- `refine_mask_v2.py`: Latest version of the mask refinement process with improved logic for combining pose-based boxes and segmentation
- `refine_mask.py`: Original implementation of the mask refinement process
- `video_to_pose_mask.py`: Handles pose estimation and creates ROI masks for hands and faces

### Detection Methods

The system uses a hybrid approach combining:

1. **YOLOv8 Segmentation**: For detailed segmentation of hands and faces
2. **Pose-Based Detection**: For reliable bounding boxes around hands and face regions
3. **Combined Logic**: Intelligently uses pose-based boxes when segmentation is absent or incomplete

## Usage

### Processing a Video

```python
python refine_mask_v2.py "path/to/video.mp4" "output_directory"
```

This will generate two output videos:
- `*_seg_mask_v2.mp4`: Video with segmentation masks overlaid
- `*_combined_mask.mp4`: Video with the combined mask approach

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- FFmpeg (for video processing)
- DWpose (for pose estimation)

## Implementation Details

The mask refinement process follows these steps:

1. Detect persons in each frame
2. For each person region:
   - Apply hand and face detection using YOLOv8
   - Extract pose-based bounding boxes for hands and face
   - Create a combined mask that:
     - Uses full YOLOv8 segmentation when it intersects with pose boxes
     - Uses pose-based boxes when no segmentation is detected
   - Constrain the combined mask to the person region

This approach ensures robust detection even when one method fails, providing more consistent results across different video conditions.
