import cv2
import numpy as np
import logging
import os
import sys
import mediapipe as mp

MP_LANDMARKS_NUM = 543  # Number of landmarks in MediaPipe Holistic model

class PoseEstimator:
    """Class for estimating pose landmarks using MediaPipe."""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True)

    def __init__(self, thread_id, path):
        self.path_to_video = os.path.join(path, f"{thread_id}.mp4")
        self.all_coords = []
        self.all_visibility = []
    
    def estimate(self):
        """Estimate pose landmarks from an image."""
        try:
            cap = cv2.VideoCapture(self.path_to_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = self.holistic.process(image)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
                    pose_visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)
                else:
                    pose_coords = np.full((33, 3), np.nan, dtype=np.float64)
                    pose_visibility = np.full((33,), np.nan, dtype=np.float64)
                if results.right_hand_landmarks:
                    landmarks = results.right_hand_landmarks.landmark
                    right_hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
                    right_hand_visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)
                else:
                    right_hand_coords = np.full((21, 3), np.nan, dtype=np.float64)
                    right_hand_visibility = np.full((21,), np.nan, dtype=np.float64)
                if results.left_hand_landmarks:
                    landmarks = results.left_hand_landmarks.landmark
                    left_hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
                    left_hand_visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)
                else:
                    left_hand_coords = np.full((21, 3), np.nan, dtype=np.float64)
                    left_hand_visibility = np.full((21,), np.nan, dtype=np.float64) 
                if results.face_landmarks:
                    landmarks = results.face_landmarks.landmark
                    face_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
                    face_visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)
                else:
                    face_coords = np.full((468, 3), np.nan, dtype=np.float64)
                    face_visibility = np.full((468,), np.nan, dtype=np.float64)
                # print(f'{pose_coords.shape=} {right_hand_coords.shape=} {left_hand_coords.shape=} {face_coords.shape=}')
                combined_coords = np.vstack((pose_coords, right_hand_coords, left_hand_coords, face_coords))
                # print(f'{combined_coords.shape=}')
                # print(f'{pose_visibility.shape=} {right_hand_visibility.shape=} {left_hand_visibility.shape=} {face_visibility.shape=}')
                combined_visibility = np.hstack((pose_visibility, right_hand_visibility, left_hand_visibility, face_visibility))
                # print(f'{combined_visibility.shape=}')
                self.all_coords.append(combined_coords)
                self.all_visibility.append(combined_visibility)
            cap.release()
        except Exception as exce:
            raise Exception("Error with video processing with mediapipe") from exce
        return np.array(self.all_coords, dtype=np.float64), np.array(self.all_visibility, dtype=np.float64)



def pose_estimate_npz(id, path):
    """Estimate pose landmarks and save to a compressed npz file."""
    try:
        pose_estim = PoseEstimator(id, path)
        coords, visibility = pose_estim.estimate()

        assert coords[0].shape == (MP_LANDMARKS_NUM,3), f"{coords.shape=}"
        assert visibility[0].shape == (MP_LANDMARKS_NUM,), f"{visibility[0].shape=}"
        assert coords.shape[0] == visibility.shape[0], f"{coords.shape=} {visibility.shape=}"

        output_path = os.path.join(path, f"{id}.pose-mediapipe.npz")    
        np.savez_compressed(output_path, frames=coords, visibility=visibility)
        logging.info(f"Saved pose landmarks to {output_path}")
    except Exception as exce:
        logging.error(f"Error in generating pose files for id {id}!")
        logging.exception("Exception occurred:")
        raise Exception(f"video {id}.mp4: Error in pose estimation and npz generation using mediapipe") from exce

if __name__ == '__main__':
  if len(sys.argv) < 2:
      logging.error("Usage: python mp_processing.py <video_id> <path>")
      sys.exit(1)
  video_id = sys.argv[1]
  path = sys.argv[2] if len(sys.argv) > 2 else "/content/isl_gospel_videos"
  pose_estimate_npz(video_id, path)
