import sys
import os
import io
from tqdm import tqdm
import matplotlib
import cv2
import torch
import numpy as np
import logging

logging.basicConfig(filename='/my_logs/logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DWPOSE_LANDMARKS_NUM = 134


from dwpose import util
from dwpose import DWposeDetector
from dwpose.wholebody import Wholebody
pose_estimation = Wholebody()

def pose_estimate_v3(oriImg):
  H, W, C = oriImg.shape
  with torch.no_grad():
      candidate, subset = pose_estimation(oriImg)
      nums, keys, locs = candidate.shape
      candidate[..., 0] /= float(W)
      candidate[..., 1] /= float(H)
      un_visible = subset<0.3
      candidate[un_visible] = -1

      return candidate, subset



def draw_pose(pose, H, W, hands_scores):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = draw_handpose(canvas, hands, hands_scores)

    canvas = util.draw_facepose(canvas, faces)

    return canvas

eps = 0.01
def draw_handpose(canvas, all_hand_peaks, hands_scores):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks,scores in zip(all_hand_peaks, hands_scores):
        peaks = np.array(peaks)
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

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

def draw_mask(pose, H, W):
    faces = pose['faces']
    hands = pose['hands']

    mask = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    mask = draw_roi_mask(mask, faces[0], H, W)
    mask = draw_roi_mask(mask, hands[0], H, W)
    mask = draw_roi_mask(mask, hands[1], H, W)
    return mask

def draw_roi_mask(mask, all_points, H, W):
    for point in all_points:
      point[0] = int(point[0]*W)
      point[1] = int(point[1]*H)
    min_x = np.min(all_points[:, 0])
    max_x = np.max(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_y = np.max(all_points[:, 1])
    points = np.array([(min_x,min_y),(min_x, max_y), (max_x, max_y), (max_x, min_y)], dtype=np.int32)
    if min_x > 0 and min_y > 0 and max_x < W and max_y < H:
        cv2.fillPoly(mask, [points], color=(255,255,255))
    return mask


def pose_estimate_v2(image):
  oriImg = image.copy()
  H, W, C = oriImg.shape
  with torch.no_grad():
      candidate, subset = pose_estimation(oriImg)
      nums, keys, locs = candidate.shape
      candidate[..., 0] /= float(W)
      candidate[..., 1] /= float(H)
      body = candidate[:,:18].copy()
      body = body.reshape(nums*18, locs)
      score = subset[:,:18]
      for i in range(len(score)):
          for j in range(len(score[i])):
              if score[i][j] > 0.3:
                  score[i][j] = int(18*i+j)
              else:
                  score[i][j] = -1

      un_visible = subset<0.3
      candidate[un_visible] = -1


      foot = candidate[:,18:24]

      faces = candidate[:,24:92]

      hands = candidate[:,92:113]
      hands = np.vstack([hands, candidate[:,113:]])

      bodies = dict(candidate=body, subset=score)
      pose = dict(bodies=bodies, hands=hands, faces=faces)

      return draw_pose(pose, H, W, hands_scores=[subset[:,92:113], subset[:, 113:]]), candidate


def pose_estimate(image):
  oriImg = image.copy()
  H, W, C = oriImg.shape
  with torch.no_grad():
      candidate, subset = pose_estimation(oriImg)
      nums, keys, locs = candidate.shape
      candidate[..., 0] /= float(W)
      candidate[..., 1] /= float(H)
      body = candidate[:,:18].copy()
      body = body.reshape(nums*18, locs)
      score = subset[:,:18]
      for i in range(len(score)):
          for j in range(len(score[i])):
              if score[i][j] > 0.3:
                  score[i][j] = int(18*i+j)
              else:
                  score[i][j] = -1

      un_visible = subset<0.3
      candidate[un_visible] = -1


      foot = candidate[:,18:24]

      faces = candidate[:,24:92]

      hands = candidate[:,92:113]
      hands = np.vstack([hands, candidate[:,113:]])

      bodies = dict(candidate=body, subset=score)
      pose = dict(bodies=bodies, hands=hands, faces=faces)

      return draw_pose(pose, H, W, hands_scores=[subset[:,92:113], subset[:, 113:]]), draw_mask(pose, H, W)


def generate_mask_and_pose_video_streams(id):
    output_stream_mask = io.BytesIO()
    output_stream_pose = io.BytesIO()
    try:
      temp_file_mask =f"./{id}_mask.mp4"
      temp_file_pose =f"./{id}_pose.mp4"

      cap = cv2.VideoCapture(f"./{id}.mp4")
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = int(cap.get(cv2.CAP_PROP_FPS))
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

      # print(f"about to trim! Width: {width}, Height: {height}, FPS: {fps}")

      fourcc="mp4v"
      fourcc_code = cv2.VideoWriter_fourcc(*fourcc)  # Codec (e.g., 'mp4v', 'XVID')
      video_writer1 = cv2.VideoWriter(temp_file_mask, fourcc_code, fps, (width, height))
      video_writer2 = cv2.VideoWriter(temp_file_pose, fourcc_code, fps, (width, height))

      while True:
          ret, frame = cap.read()
          if not ret:
              break
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False

          skeleton, mask = pose_estimate(image)

          # Write frame to video
          video_writer1.write(mask)
          video_writer2.write(skeleton)
      video_writer1.release()
      video_writer2.release()
      cap.release()

      with open(temp_file_mask, "rb") as f:
        output_stream_mask.write(f.read())
      with open(temp_file_pose, "rb") as f:
          output_stream_pose.write(f.read())
    finally:
      os.remove(temp_file_mask)
      os.remove(temp_file_pose)

    output_stream_mask.seek(0);
    output_stream_pose.seek(0);
    return output_stream_mask, output_stream_pose

def generate_mask_and_pose_video_files(id):
    try:
      temp_file_mask =f"./{id}_mask.mp4"
      temp_file_pose =f"./{id}_pose.mp4"

      cap = cv2.VideoCapture(f"./{id}.mp4")
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = int(cap.get(cv2.CAP_PROP_FPS))
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

      # print(f"about to trim! Width: {width}, Height: {height}, FPS: {fps}")

      fourcc="mp4v"
      fourcc_code = cv2.VideoWriter_fourcc(*fourcc)  # Codec (e.g., 'mp4v', 'XVID')
      video_writer1 = cv2.VideoWriter(temp_file_mask, fourcc_code, fps, (width, height))
      video_writer2 = cv2.VideoWriter(temp_file_pose, fourcc_code, fps, (width, height))

      while True:
          ret, frame = cap.read()
          if not ret:
              break
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False

          skeleton, mask = pose_estimate(image)

          # Write frame to video
          video_writer1.write(mask)
          video_writer2.write(skeleton)
      video_writer1.release()
      video_writer2.release()
      cap.release()
    except Exception as exce:
        logging.error(f"Error in generating mask and pose video files for id {id}: {exce}")
        raise Exception("Error in pose and mask generation using dwpose")

def generate_pose_files_v2(id):
    try:
      # temp_file_pose =f"./{id}_pose-animation.mp4"

      cap = cv2.VideoCapture(f"/my_data/{id}.mp4")
      # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      # fps = int(cap.get(cv2.CAP_PROP_FPS))
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


      # fourcc="mp4v"
      # fourcc_code = cv2.VideoWriter_fourcc(*fourcc)  # Codec (e.g., 'mp4v', 'XVID')
      # video_writer2 = cv2.VideoWriter(temp_file_pose, fourcc_code, fps, (width, height))
      poses = []
      confidences = []

      while True:
          ret, frame = cap.read()
          if not ret:
              break
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False

          candidate, confidence = pose_estimate_v3(image)
          if len(candidate) < 1:
            candidate = [np.full(DWPOSE_LANDMARKS_NUM, np.nan, dtype=np.float64)]
          elif len(candidate > 1):
            candidate = [candidate[0]]


          if confidence.shape[0] < 1:
            confidence = np.full((1, DWPOSE_LANDMARKS_NUM), np.nan)
          elif confidence.shape[0] > 1:
            confidence = confidence[:1]

          assert len(candidate) == 1, f"{len(candidate)=}"
          assert len(candidate[0]) == DWPOSE_LANDMARKS_NUM, f"{len(candidate[0])=}"
          poses.append(candidate)

          assert confidence.shape == (1, DWPOSE_LANDMARKS_NUM), confidence.shape
          confidences.append(confidence)
          # Write frame to video
          # video_writer2.write(skeleton)
      # video_writer2.release()
      np.savez_compressed(f"/my_data/{id}.pose-dwpose.npz",
          frames=np.array(poses, dtype=np.float64),
          confidences = np.array(confidences, dtype=np.float64))
      cap.release()
    except Exception as exce:
        logging.error(f"Error in generating pose files for id {id}!")
        logging.exception("Exception occurred:")
        raise Exception("Error in pose video and array generation using dwpose") from exce

if __name__ == '__main__':
  if len(sys.argv) < 2:
      logging.error("Usage: python dwpose_processing.py <video_id>")
      sys.exit(1)
  video_id = sys.argv[1]
  generate_pose_files_v2(video_id)
