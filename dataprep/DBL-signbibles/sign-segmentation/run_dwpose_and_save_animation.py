import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env
load_dotenv()

local_folder = os.getenv("DWPOSE_PATH")
print(f"DWPOSE PATH: {local_folder}")
if local_folder not in sys.path:
    sys.path.append(local_folder)

from annotator.dwpose import util  # noqa: E402
from annotator.dwpose.wholebody import Wholebody  # noqa: E402

pose_estimator = Wholebody(
    onnx_det=Path(local_folder) / "annotator/ckpts/yolox_l.onnx",
    onnx_pose=Path(local_folder) / "annotator/ckpts/dw-ll_ucoco_384.onnx",
)


def draw_handpose(canvas, all_hand_peaks, hands_scores, eps=0.01):
    # print("Drawing Hands")
    # print(f"All hands peaks: {all_hand_peaks}")
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks, scores in zip(all_hand_peaks, hands_scores, strict=False):
        peaks = np.array(peaks)
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if scores[0][i] > 0.8:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
                elif scores[0][i] > 0.6:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 150), thickness=-1)
                elif scores[0][i] > 0.4:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 100), thickness=-1)
                else:
                    cv2.circle(canvas, (x, y), 4, (0, 0, 50), thickness=-1)
    return canvas


def draw_pose(pose: dict, height: int, width: int, hand_scores: list[np.ndarray]) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, pose["bodies"]["candidate"], pose["bodies"]["subset"])
    canvas = draw_handpose(canvas, pose["hands"], hand_scores)
    canvas = util.draw_facepose(canvas, pose["faces"])
    return canvas


def pose_estimate_with_candidates(image: np.ndarray):
    oriImg = image.copy()
    H, W, _ = oriImg.shape

    with torch.no_grad():
        keypoints, scores = pose_estimator(oriImg)
        candidate_keypoints = keypoints.copy()
        subset_scores = scores.copy()

        # Normalize coordinates to [0, 1]
        candidate_keypoints[..., 0] /= float(W)
        candidate_keypoints[..., 1] /= float(H)

        body = candidate_keypoints[:, :18].copy()
        body = body.reshape(candidate_keypoints.shape[0] * 18, candidate_keypoints.shape[2])

        score = subset_scores[:, :18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1

        un_visible = subset_scores < 0.3
        candidate_keypoints[un_visible] = -1

        foot = candidate_keypoints[:, 18:24]
        faces = candidate_keypoints[:, 24:92]
        hands = np.vstack([candidate_keypoints[:, 92:113], candidate_keypoints[:, 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        # Draw visualization
        # canvas = util.draw_bodypose(np.zeros((H, W, 3), dtype=np.uint8), body, score)
        # canvas = util.draw_facepose(canvas, faces)
        # canvas = canvas  # (skip hand drawing for now if you want)
        canvas = draw_pose(pose, H, W, hand_scores=[subset_scores[:, 92:113], subset_scores[:, 113:]])

        return canvas, keypoints, scores


def run_pose_and_save(video_path: Path, output_dir: Path, overwrite=False) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem
    animation_path = output_dir / f"{video_stem}.pose-animation.mp4"
    pose_npz_path = output_dir / f"{video_stem}.pose-dwpose.npz"
    if animation_path.is_file() and pose_npz_path.is_file() and not overwrite:
        # print(f"Already done with {pose_npz_path} and {animation_path}")
        return
        try:
            with np.load(pose_npz_path) as data:
                if "frames" in data and "confidences" in data:
                    return  # Everything exists and is complete, skip
                else:
                    print(f"No confidence values in {pose_npz_path}!")
        except (OSError, ValueError) as e:
            print(f"Warning: Could not load {pose_npz_path}: {e}")
            # fall through to rerun

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(animation_path), fourcc, fps, (width, height))

    poses = []
    confidences = []
    iterator = range(total_frames)
    if total_frames > 1000:
        iterator = tqdm(iterator, desc=f"Processing {video_stem}.", unit="frame")

    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        skeleton_frame, candidate, confidence = pose_estimate_with_candidates(rgb_frame)

        if candidate.shape[0] < 1:
            candidate = np.full((1, 134, 2), np.nan)
        elif candidate.shape[0] > 1:
            candidate = candidate[:1]

        if confidence.shape[0] < 1:
            confidence = np.full((1, 134), np.nan)
        elif confidence.shape[0] > 1:
            confidence = confidence[:1]

        assert candidate.shape == (1, 134, 2), f"Unexpected shape: {candidate.shape}"
        poses.append(candidate)

        assert confidence.shape == (1, 134), confidence.shape
        confidences.append(confidence)

        writer.write(skeleton_frame)

    writer.release()
    cap.release()

    # Convert lists to arrays with appropriate dtype
    frames_array = np.array(poses, dtype=np.float64)  # Shape: (N, 1, 134, 2)
    conf_array = np.array(confidences, dtype=np.float64)  # Shape: (N, 1, 134)

    # Save both arrays into a compressed .npz file
    np.savez_compressed(pose_npz_path, frames=frames_array, confidences=conf_array)
    print(f"Saved pose animation to {animation_path}")
    print(f"Saved pose data to {pose_npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DWPose and save results.")
    parser.add_argument("video_path", type=Path, help="Input video path or directory")
    parser.add_argument("--output_dir", type=Path, help="Directory to save outputs")

    args = parser.parse_args()

    if args.video_path.is_dir():
        # TODO: recursive arg
        video_paths = list(args.video_path.rglob("*.mp4"))
        video_paths = [p for p in video_paths if "segments" not in p.parents]
        video_paths = [p for p in video_paths if "_segment_" not in p.name]
        video_paths = [p for p in video_paths if "segment_" not in p.name]
        video_paths = [p for p in video_paths if "_SIGN_" not in p.name]
        video_paths = [p for p in video_paths if "_SENTENCE_" not in p.name]
        video_paths = [p for p in video_paths if ".pose-animation.mp4" not in p.name]

    else:
        video_paths = [args.video_path]
    print(f"{len(video_paths)} videos to process")
    for video_path in tqdm(video_paths, desc="Processing videos with DWPose"):
        if args.output_dir is None:
            output_dir = video_path.parent
        else:
            output_dir = args.output_dir
        run_pose_and_save(video_path, output_dir)
# GPU
# conda activate /opt/home/cleong/envs/onnxruntime_gpu/ && python /opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/dataprep/DBL-signbibles/sign-segmentation/run_dwpose_and_save_animation.py downloads/bqn
