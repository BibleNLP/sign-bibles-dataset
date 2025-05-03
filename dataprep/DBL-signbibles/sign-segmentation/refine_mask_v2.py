import cv2
import numpy as np
import os
import subprocess
from autodistill_yolov8 import YOLOv8
import torch
import sys
from annotator.dwpose import DWposeDetector
from video_to_pose_mask import pose_estimate, draw_roi_mask

# Initialize DWpose detector
dwpose = DWposeDetector()

class PersonDetector:
    def __init__(self, model_path):
        self.model = YOLOv8(model_path)
    
    def predict(self, frame):
        """
        Predict person location using YOLOv8
        Returns the bounding box of the person detection with highest confidence
        """
        results = self.model.predict(frame)[0]
        
        # Find person detection with highest confidence
        best_box = None
        best_conf = -1
        
        if results.boxes is not None:
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.cpu().numpy()
        
        return best_box

class HandFaceDetector:
    def __init__(self, model_path):
        self.model = YOLOv8(model_path)
    
    def predict(self, frame):
        """
        Predict using YOLOv8 for detection and segmentation
        Returns both the segmentation mask and a list of bounding boxes
        """
        results = self.model.predict(frame)[0]
        
        # Create a blank mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Process segmentation masks
        has_mask = False
        if results.masks is not None and len(results.masks.data) > 0:
            has_mask = True
            for segment in results.masks.data:
                # Convert the segment to binary mask
                segment = segment.cpu().numpy()
                segment = (segment > 0.5).astype(np.uint8) * 255
                # Resize segment to match frame size
                segment = cv2.resize(segment, (frame.shape[1], frame.shape[0]))
                mask = cv2.bitwise_or(mask, segment)
        
        # Get bounding boxes
        boxes = []
        if results.boxes is not None:
            for box in results.boxes.xyxy:
                boxes.append(box.cpu().numpy().astype(int))
        
        return mask, boxes, has_mask

def expand_box(box, frame_shape, expand_percent=0.2):
    """Expand a bounding box by a percentage while keeping it within frame bounds"""
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    
    # Calculate expansion amount
    expand_x = int(width * expand_percent / 2)
    expand_y = int(height * expand_percent / 2)
    
    # Expand the box
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = min(frame_shape[1], x2 + expand_x)
    new_y2 = min(frame_shape[0], y2 + expand_y)
    
    return [new_x1, new_y1, new_x2, new_y2]

def extract_square_region(frame, box, padding_factor=0.1):
    """Extract a square region around the detection box with padding"""
    x1, y1, x2, y2 = map(int, box)
    
    # Calculate center and size
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    size = max(x2 - x1, y2 - y1)
    
    # Add padding
    size = int(size * (1 + padding_factor))
    
    # Calculate new coordinates ensuring they're within frame bounds
    new_x1 = max(0, center_x - size // 2)
    new_y1 = max(0, center_y - size // 2)
    new_x2 = min(frame.shape[1], center_x + size // 2)
    new_y2 = min(frame.shape[0], center_y + size // 2)
    
    # Extract region
    region = frame[new_y1:new_y2, new_x1:new_x2]
    
    return region, (new_x1, new_y1, new_x2, new_y2)

def create_box_mask(frame_shape, box):
    """Create a binary mask from a bounding box"""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    mask[y1:y2, x1:x2] = 255
    return mask

def check_box_overlap(box1, box2):
    """Check if two boxes overlap"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check if one box is to the left of the other
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    
    # Check if one box is above the other
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    
    # Boxes overlap
    return True

def get_pose_boxes(frame):
    """
    Get hand and face bounding boxes using pose estimation
    Returns a list of boxes in [x1, y1, x2, y2] format
    """
    H, W = frame.shape[:2]
    
    # Run pose estimation
    _, mask_img = pose_estimate(frame)
    
    # Convert mask to grayscale
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes from contours
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])
    
    # Expand boxes by 20%
    expanded_boxes = []
    for box in boxes:
        expanded_boxes.append(expand_box(box, (H, W), 0.2))
    
    return expanded_boxes

def process_video(video_path, output_dir):
    """Process a video file using two-stage YOLOv8 detection.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the output
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Initialize the models
    person_model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8_person-seg.pt")
    hand_face_model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8_hand_face-seg.pt")
    
    person_detector = PersonDetector(person_model_path)
    hand_face_detector = HandFaceDetector(hand_face_model_path)
    
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
    mask_output = os.path.join(output_dir, f"{base_name}_seg_mask_v2.mp4")
    combined_mask_output = os.path.join(output_dir, f"{base_name}_combined_mask.mp4")
    
    # Set up ffmpeg process for mask video
    ffmpeg_mask = subprocess.Popen([
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # Input from pipe
        '-c:v', 'libx264',
        '-profile:v', 'baseline',  # Use baseline profile for better compatibility
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',  # Standard pixel format for better compatibility
        '-movflags', '+faststart',  # Optimize for web streaming
        mask_output
    ], stdin=subprocess.PIPE)
    
    # Set up ffmpeg process for combined mask video
    ffmpeg_combined = subprocess.Popen([
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # Input from pipe
        '-c:v', 'libx264',
        '-profile:v', 'baseline',  # Use baseline profile for better compatibility
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',  # Standard pixel format for better compatibility
        '-movflags', '+faststart',  # Optimize for web streaming
        combined_mask_output
    ], stdin=subprocess.PIPE)
    
    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Create empty masks for this frame
            final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # First stage: detect person
            person_box = person_detector.predict(frame)
            
            if person_box is not None:
                # Extract square region around person
                region, coords = extract_square_region(frame, person_box)
                
                # Second stage: detect hands and face in the region
                region_mask, region_boxes, has_region_mask = hand_face_detector.predict(region)
                
                # Place the mask back in the original frame position
                x1, y1, x2, y2 = coords
                region_h, region_w = region_mask.shape
                final_mask[y1:y1+region_h, x1:x1+region_w] = region_mask
                
                # Adjust YOLO boxes to the original frame coordinates
                adjusted_boxes = []
                for box in region_boxes:
                    # Adjust box coordinates to the original frame
                    adj_box = [
                        box[0] + x1, box[1] + y1,
                        box[2] + x1, box[3] + y1
                    ]
                    # Expand the box by 20%
                    adj_box = expand_box(adj_box, frame.shape, 0.2)
                    adjusted_boxes.append(adj_box)
                
                # Always get pose-based boxes
                pose_boxes = get_pose_boxes(frame)
                
                # Combine all boxes
                all_boxes = pose_boxes
                
                # If no boxes were detected at all, use the person box as a fallback
                if len(all_boxes) == 0:
                    all_boxes = [person_box]
                
                # Create a mask for each box
                box_masks = []
                for box in all_boxes:
                    box_mask = create_box_mask(frame.shape, box)
                    box_masks.append((box, box_mask))
                
                # For each box, check if we should use the box or the segmentation
                combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                
                # First, add all segmentation from YOLO that intersects with any pose box
                for box, box_mask in box_masks:
                    # Check if there's any segmentation in this box
                    box_segmentation = cv2.bitwise_and(final_mask, box_mask)
                    
                    # If there's any segmentation that intersects with this box, add the full segmentation
                    if np.sum(box_segmentation) > 0:
                        # Add the full segmentation to the combined mask
                        combined_mask = cv2.bitwise_or(combined_mask, final_mask)
                    
                # Now add any pose boxes that don't have segmentation in them
                for box, box_mask in box_masks:
                    # Check if there's any segmentation in this box
                    box_segmentation = cv2.bitwise_and(final_mask, box_mask)
                    
                    # If there's no segmentation in this box, use the box
                    if np.sum(box_segmentation) == 0:
                        combined_mask = cv2.bitwise_or(combined_mask, box_mask)
                
                # Ensure the combined mask is within the person region
                person_mask = create_box_mask(frame.shape, person_box)
                combined_mask = cv2.bitwise_and(combined_mask, person_mask)
            
            # Convert masks to 3-channel for video writing
            mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            combined_mask_3ch = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            
            # Write frames to ffmpeg processes
            ffmpeg_mask.stdin.write(mask_3ch.tobytes())
            ffmpeg_combined.stdin.write(combined_mask_3ch.tobytes())
            
    finally:
        cap.release()
        # Close ffmpeg processes
        ffmpeg_mask.stdin.close()
        ffmpeg_mask.wait()
        ffmpeg_combined.stdin.close()
        ffmpeg_combined.wait()
        
    print(f"Processed video saved to: {mask_output}")
    print(f"Combined mask video saved to: {combined_mask_output}")
    
    # Return True to indicate success
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video using two-stage YOLOv8 detection')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('output_dir', help='Directory to save the output')
    
    args = parser.parse_args()
    process_video(args.video_path, args.output_dir)
