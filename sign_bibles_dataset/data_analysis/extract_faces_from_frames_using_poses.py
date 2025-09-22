import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from pose_format import Pose



def setup_logging(level: int = logging.INFO):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=level)


def expand_bbox_by_percent(x1, y1, x2, y2, percent, img_width, img_height):
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * percent
    pad_y = box_height * percent
    x1_new = max(int(x1 - pad_x), 0)
    y1_new = max(int(y1 - pad_y), 0)
    x2_new = min(int(x2 + pad_x), img_width - 1)
    y2_new = min(int(y2 + pad_y), img_height - 1)
    return x1_new, y1_new, x2_new, y2_new


def get_face_crop_for_frame(
    video_path: Path, pose_path: Path, frame_index: int, expand_percent: float = 0.0
) -> list[np.ndarray]:
    """
    Return a list of cropped face regions from a single frame.

    Returns:
        * A list of cropped face regions (np.ndarray), one per person.
        * the frame itself
        Skips masked bboxes.


    """
    video_path = Path(video_path)
    pose_path = Path(pose_path)

    pose = Pose.read(pose_path.read_bytes())
    face_pose = pose.get_components(["FACE_LANDMARKS"])
    pose_bbox = pose.bbox()
    face_bbox = face_pose.bbox()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Seek to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read frame {frame_index} from {video_path}")

    img_height, img_width = frame.shape[:2]
    frame_bboxes = face_bbox.body.data[frame_index]
    full_person_bboxes = pose_bbox.body.data[frame_index]

    crops = []
    for person, bbox in enumerate(frame_bboxes):
        if np.ma.is_masked(bbox) or bbox.mask.any():
            logging.debug(f"Frame {frame_index}, person {person}: bbox is masked, skipping.")
            continue

        upper_left = bbox[0]
        bottom_right = bbox[1]

        ul_x, ul_y, br_x, br_y = expand_bbox_by_percent(
            upper_left[0],
            upper_left[1],
            bottom_right[0],
            bottom_right[1],
            percent=expand_percent,
            img_width=img_width,
            img_height=img_height,
        )

        crop = frame[ul_y:br_y, ul_x:br_x]
        crops.append(crop)

    return crops, frame


def main():
    parser = argparse.ArgumentParser(description="Extract face crops from pose and video for each frame.")
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Path to the directory containing frame_*.png images (e.g. samples/CBT-001-..._frames)",
    )
    parser.add_argument("--expand", type=float, default=0.3, help="Percentage to expand face bounding box")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    frames_dir: Path = args.frames_dir
    expand_percent = args.expand

    if not frames_dir.is_dir():
        raise ValueError(f"{frames_dir} is not a directory")

    base_path = frames_dir.parent / frames_dir.name.replace("_frames", "")
    video_path = base_path.with_suffix(".mp4")
    pose_path = base_path.with_suffix(".pose")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    facecrop_dir = frames_dir / "facecrops"
    people_dir = frames_dir / "frames_with_people"
    
    facecrop_dir.mkdir(exist_ok=True)
    people_dir.mkdir(exist_ok=True)

    frame_pattern = re.compile(r"frame_(\d+)\.png")
    frame_files = sorted(frames_dir.glob("frame_*.png"))

    for frame_file in frame_files:
        match = frame_pattern.match(frame_file.name)
        if not match:
            logging.debug(f"Skipping non-matching file: {frame_file}")
            continue

        frame_idx = int(match.group(1))

        crops, frame = get_face_crop_for_frame(
            video_path=video_path,
            pose_path=pose_path,
            frame_index=frame_idx,
            expand_percent=expand_percent,
        )

        if crops:
            person_frame_filename = people_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(person_frame_filename, frame)

        for i, crop in enumerate(crops):
            crop_filename = facecrop_dir / f"frame_{frame_idx:05d}_person_{i}.png"
            cv2.imwrite(str(crop_filename), crop)
            logging.debug(f"Saved: {crop_filename}")

    crops_saved_count = len(list(facecrop_dir.glob("frame_*person*.png")))
    logging.info(f"There are now {crops_saved_count} face crops in {facecrop_dir.resolve()}")
        


if __name__ == "__main__":
    main()
