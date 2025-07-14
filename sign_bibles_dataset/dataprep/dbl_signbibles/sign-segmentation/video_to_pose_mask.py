import sys
import os
import subprocess
import torch
from annotator.dwpose import DWposeDetector, util
from annotator.dwpose.wholebody import Wholebody
import matplotlib.colors
import cv2
import numpy as np
import argparse

# Add the dwpose directory to the Python path
dwpose_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dwpose")
sys.path.append(dwpose_dir)

dwpose = DWposeDetector()


def draw_pose(pose, H, W, hands_scores=None, face_scores=None):
    """Draw full body pose including body, hands, and face."""
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = draw_handpose(canvas, hands, hands_scores)
    canvas = draw_facepose(canvas, faces, face_scores)  # Updated call

    return canvas


def draw_handpose(canvas, all_hand_peaks, hands_scores):
    """Draw hand keypoints and connections."""
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

    eps = 0.01
    for peaks, scores in zip(all_hand_peaks, hands_scores):
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
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=2,
                )

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
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


def draw_facepose(canvas, all_face_peaks, scores=None):
    """Draw face keypoints on canvas."""
    if len(all_face_peaks) == 0:
        return canvas

    H, W = canvas.shape[:2]  # Get canvas dimensions
    for i, face_peaks in enumerate(all_face_peaks):
        for j, peak in enumerate(face_peaks):
            if peak[0] < 0 or peak[1] < 0:  # Skip invalid points
                continue

            # Scale coordinates back to image dimensions
            x, y = int(peak[0] * W), int(peak[1] * H)
            if scores is not None:
                # Scale brightness based on confidence score (0.25-1.0 range)
                brightness = 0.25 + (0.75 * scores[j]) if j < len(scores) else 0.25
                color = (
                    int(255 * brightness),
                    int(255 * brightness),
                    int(255 * brightness),
                )
            else:
                color = (255, 255, 255)

            # Draw circle at x,y
            cv2.circle(canvas, (x, y), 2, color, thickness=2)

    return canvas


def draw_mask(pose, H, W):
    """Draw ROI masks for face and hands."""
    faces = pose["faces"]
    hands = pose["hands"]

    mask = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    mask = draw_roi_mask(mask, faces[0], H, W)
    mask = draw_roi_mask(mask, hands[0], H, W)
    mask = draw_roi_mask(mask, hands[1], H, W)
    return mask


def draw_roi_mask(mask, all_points, H, W):
    """Draw ROI mask for a set of points."""
    for point in all_points:
        point[0] = int(point[0] * W)
        point[1] = int(point[1] * H)
    min_x = np.min(all_points[:, 0])
    max_x = np.max(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_y = np.max(all_points[:, 1])
    points = np.array(
        [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)], dtype=np.int32
    )
    cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask


def pose_estimate(oriImg):
    """Estimate pose from image."""
    H, W, C = oriImg.shape
    with torch.no_grad():
        candidate, subset = dwpose.pose_estimation(oriImg)
        nums, keys, locs = candidate.shape

        # Scale coordinates to 0-1 range first
        candidate[..., 0] /= float(W)  # Scale x coordinates to 0-1
        candidate[..., 1] /= float(H)  # Scale y coordinates to 0-1

        # Prepare body keypoints
        body = candidate[:, :18].copy()
        body = body.reshape(nums * 18, locs)
        score = subset[:, :18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1

        # Mark points with low confidence as invisible
        un_visible = subset < 0.3
        candidate[un_visible] = -1

        # Extract different body parts
        foot = candidate[:, 18:24]
        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])

        # Create pose dictionary with scaled coordinates
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        # Extract face scores and normalize them to 0-1 range
        face_scores = None
        if len(subset) > 0:
            face_scores = subset[0, 24:92].copy()  # Get face keypoint scores
            # Normalize scores to 0-1 range and ensure they're above minimum threshold
            face_scores = np.clip((face_scores - 0.3) / 0.7, 0, 1)  # Map 0.3-1.0 to 0-1
            face_scores = np.where(
                face_scores < 0.2, 0, face_scores
            )  # Set very low scores to 0

        # Draw pose and mask
        pose_img = draw_pose(
            pose,
            H,
            W,
            hands_scores=[subset[:, 92:113], subset[:, 113:]],
            face_scores=face_scores,
        )
        mask_img = draw_mask(pose, H, W)
        return pose_img, mask_img


def process_video(input_path, output_dir):
    """Process video file to generate pose and mask visualizations."""
    # Initialize DWPose model
    pose_model = Wholebody()

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Set up ffmpeg processes for pose and mask videos
    pose_output = os.path.join(output_dir, f"{base_name}_pose.mp4")
    mask_output = os.path.join(output_dir, f"{base_name}_mask.mp4")

    # Get codec information from the original video
    fourcc_str = ""
    try:
        # Try to get the codec from the original video
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_chars = [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
        fourcc_str = "".join(fourcc_chars)
        print(f"Original video codec: {fourcc_str}")
    except:
        # Default to H.264 if we can't get the original codec
        fourcc_str = "avc1"
        print(f"Using default codec: {fourcc_str}")

    ffmpeg_pose = subprocess.Popen(
        [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",  # Input from pipe
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",  # Use baseline profile for better compatibility
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",  # Standard pixel format for better compatibility
            "-movflags",
            "+faststart",  # Optimize for web streaming
            pose_output,
        ],
        stdin=subprocess.PIPE,
    )

    ffmpeg_mask = subprocess.Popen(
        [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",  # Input from pipe
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",  # Use baseline profile for better compatibility
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",  # Standard pixel format for better compatibility
            "-movflags",
            "+faststart",  # Optimize for web streaming
            mask_output,
        ],
        stdin=subprocess.PIPE,
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with DWPose
        pose_frame, mask_frame = pose_estimate(frame)

        # Write frames to ffmpeg processes
        ffmpeg_pose.stdin.write(pose_frame.tobytes())
        ffmpeg_mask.stdin.write(mask_frame.tobytes())

        frame_count += 1
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")

    print("\nDone!")

    # Close ffmpeg processes
    ffmpeg_pose.stdin.close()
    ffmpeg_mask.stdin.close()
    ffmpeg_pose.wait()
    ffmpeg_mask.wait()

    # Release resources
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Process video with DWPose")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()

    process_video(args.input, args.output_dir)


if __name__ == "__main__":
    main()
