import cv2
import mediapipe as mp
import os
import sys
import io

class VideoTrimmer:
    """Class for trimming a video stream by detecting the first frame with a person and a face."""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True)

    def __init__(self, id):
        """Initialize with a video source (file path or camera index)."""
        # print("comes in videotrimmer init")
        self.thread_id = id

    def find_trim_frame_opencv(self):
        """Find the first frame where both a person and a face are visible."""
        frame_idx = 0
        trim_frame = None

        try:
          temp_file1 = f"./{self.thread_id}.mp4"
          cap = cv2.VideoCapture(temp_file1)
          right_hand_pos = None
          left_hand_pos = None
          hand_displacement = 0

          while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with Holistic model
            results = self.holistic.process(rgb_frame)

            person_detected = results.pose_landmarks is not None
            if not person_detected or not results.pose_landmarks.landmark[12].x:
              continue
            # face_detected = results.face_landmarks is not None
            if results.right_hand_landmarks:
              if right_hand_pos is None:
                right_hand_pos = results.right_hand_landmarks.landmark[16].y
              else:
                hand_displacement = max(hand_displacement,
                                        abs(right_hand_pos-results.right_hand_landmarks.landmark[16].y))

            if results.left_hand_landmarks:
              if left_hand_pos is None:
                left_hand_pos = results.left_hand_landmarks.landmark[15].y
              else:
                hand_displacement = max(hand_displacement,
                                        abs(left_hand_pos-results.left_hand_landmarks.landmark[15].y))
            # print(f'{frame_idx=} {hand_displacement=}')

            if hand_displacement>0.1:
                trim_frame = max(frame_idx-10, 0)
                break
          cap.release()
          cv2.destroyAllWindows()
        except Exception as exce:
          raise Exception("Error with video processing with mediapipe") from exce

        return trim_frame

    def trim_video(self):
        """Trims the video stream and returns an OpenCV VideoCapture-like output stream."""
        # print("trimming...")
        trim_frame = self.find_trim_frame_opencv()

        if trim_frame is None:
            print("No valid trim frame found. Returning None.")
            return None
        # print(f"Trim frame found at frame {trim_frame}.")
        if trim_frame < 10:
          print(f"Trim frame {trim_frame} < 10. Hence keeping the video as such")
          return True

        # trim_frame += 10

        try:
          temp_file2 =f"./{self.thread_id}_trimmed.mp4"

          cap = cv2.VideoCapture(f"./{self.thread_id}.mp4")
          width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          fps = int(cap.get(cv2.CAP_PROP_FPS))
          cap.set(cv2.CAP_PROP_POS_FRAMES, trim_frame)

          # print(f"about to trim! Width: {width}, Height: {height}, FPS: {fps}")

          fourcc="avc1"
          fourcc_code = cv2.VideoWriter_fourcc(*fourcc)  # Codec (e.g., 'mp4v', 'XVID')
          video_writer = cv2.VideoWriter(temp_file2, fourcc_code, fps, (width, height))

          while True:
              ret, frame = cap.read()
              if not ret:
                  break

              # Write frame to video
              video_writer.write(frame)
          video_writer.release()
          cap.release()

        finally:
          os.remove(f"./{self.thread_id}.mp4")
          os.rename(temp_file2, f"./{self.thread_id}.mp4")
          # self.stream.seek(0)

        return True

# Function for parallel processing
def trim_off_storyboard(id):
    """Creates a VideoTrimmer object and processes a single video."""
    # print("Comes in trimmer!")
    trimmer = VideoTrimmer(id)
    # print("Trimming!")
    return trimmer.trim_video()
