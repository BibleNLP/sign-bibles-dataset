# HuggingFace Preparation for Sign Language Videos

This module prepares sign language videos from DBL-sign for use with HuggingFace datasets. It:

1. Downloads videos from DBL-sign
2. Processes them with sign-segmentation to extract pose, segmentation, and mask data
3. Packages everything into WebDataset format

## Features

- Download X number of videos from DBL-sign
- Process videos to extract segments, pose data, and segmentation masks
- Create WebDataset format with .pose, .mask, .segmentation, and .original files
- Include copyright and permission data from DBL-sign
- Track segment timing information
- Interactive GUI progress display for monitoring processing

## Usage

```bash
# Basic usage
python prepare_webdataset.py --num-videos 10 --output-dir ./output

# With specific language filter
python prepare_webdataset.py --num-videos 10 --language-code ASL --output-dir ./output

# With specific project filter
python prepare_webdataset.py --num-videos 10 --project-name "ASLN" --output-dir ./output

# With GUI progress display
python prepare_webdataset.py --num-videos 10 --output-dir ./output --with-gui

# Alternative: Use the dedicated GUI script
python run_with_gui.py --num-videos 10 --language sqs
```

## Progress GUI

The progress GUI provides real-time monitoring of the video processing pipeline:

- Overall progress tracking across all processing steps
- Detailed per-operation progress updates
- Processing log with timestamps
- Pause/Resume and Cancel functionality
- Visual indicators for current operations

To use the GUI, either:
1. Add the `--with-gui` flag to the `prepare_webdataset.py` command
2. Run the dedicated `run_with_gui.py` script

## WebDataset Format

Each segment is stored as a group of files with the same base name but different extensions:
- `.original.mp4`: Original video segment
- `.pose.mp4`: Video with pose keypoints visualization
- `.mask.mp4`: Video with segmentation masks
- `.segmentation.mp4`: Video with combined segmentation
- `.json`: Metadata including copyright, permissions, and segment timing

## Requirements

See `requirements.txt` for all dependencies.

## License

This code is provided under the same license as the signbibles-processing repository.
