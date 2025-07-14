import cv2
import numpy as np
import os
import subprocess
from autodistill_yolov8 import YOLOv8


class HandFaceDetector:
    def __init__(self, model_path):
        # Initialize YOLOv8 model
        self.model = YOLOv8(model_path)

    def predict(self, frame):
        """
        Predict using YOLOv8 for detection and segmentation
        """
        # Get detections from YOLOv8
        results = self.model.predict(frame)[0]

        # Create a blank mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Process detections
        if results.masks is not None:
            for segment in results.masks.data:
                # Convert the segment to binary mask
                segment = segment.cpu().numpy()
                segment = (segment > 0.5).astype(np.uint8) * 255
                # Resize segment to match frame size
                segment = cv2.resize(segment, (frame.shape[1], frame.shape[0]))
                mask = cv2.bitwise_or(mask, segment)

        return mask


def process_video(video_path, output_dir):
    """Process a video file using YOLOv8 for detection."""
    # Initialize the model with the local path
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "yolov8_hand_face-seg.pt"
    )
    model = HandFaceDetector(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    mask_output = os.path.join(output_dir, f"{base_name}_seg_mask.mp4")

    # Set up ffmpeg process for mask video
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
            "-preset",
            "medium",
            "-crf",
            "23",
            mask_output,
        ],
        stdin=subprocess.PIPE,
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get mask from model
            mask = model.predict(frame)

            # Convert mask to 3-channel for video writing
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Write frame to ffmpeg process
            ffmpeg_mask.stdin.write(mask_3ch.tobytes())

    finally:
        cap.release()
        # Close ffmpeg process
        ffmpeg_mask.stdin.close()
        ffmpeg_mask.wait()

    print(f"Processed video saved to: {mask_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video using YOLOv8")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("output_dir", help="Directory to save the output")

    args = parser.parse_args()
    process_video(args.video_path, args.output_dir)
