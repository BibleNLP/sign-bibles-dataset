import json
import os
import io
import glob
import sys
import re
from pathlib import Path
import shutil
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import ffmpeg
import datetime
import logging

from ffmpeg_downsample import downsample_video_ondisk
from pose_format_util import video2poseformat
from mediapipe_trim import trim_off_storyboard
from pose_format_util import video2poseformat
# from dwpose_processing import generate_pose_files_v2

# Load environment variables from .env
load_dotenv()

logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_video(id, input_path, output_path):
	"""Pipeline:
		--> input: video file in local path, with name indicating the gloss
		--> downsample the video
		--> process: pose estimation, metadata etc
		--> keep in a local path for tar building later"""
	main_path = Path(output_path) / f"{id}.mp4"
	try:
		shutil.copy(input_path, f"./{id}_large.mp4") 
		downsample_video_ondisk(f"{id}_large.mp4", f"{id}.mp4")
		

		trimmed_stream = trim_off_storyboard(None, id)
		if not trimmed_stream:
			raise Exception("Processing with mediapipe failed")
		video2poseformat(id) #  .pose format using mediapipe


		shutil.move(f"{id}.mp4", main_path)
		shutil.move(f"{id}_pose-mediapipe.pose", f"{output_path}/{id}.pose-mediapipe.pose")

		text = input_path.split("/")[-1].split(".")[0].lower()
		if text.endswith("-") or text.endswith(" "):
			text = text[:-1]	

		probe = ffmpeg.probe(main_path)
		duration = float(probe['format']['duration'])
		width = 0
		height = 0
		fps = 0
		for stream in probe['streams']:
			if stream['codec_type'] == 'video': 
				width = int(stream['width'])
				height = int(stream['height'])
				fps_str = stream['r_frame_rate']      # e.g., '25/1'
				num, denom = map(int, fps_str.split('/'))
				fps = num / denom

				break

		frame_count = int(duration * fps)

		metadata = {
					"language": {
						"name": "Indian Sign Language",
						"nameLocal" : "Indian Sign Language",
						"ISO639-3": "ins",
						"BCP-47": "ins-IN"
					},
					"project": "Indian Sign Language Bible (ISLV) Dictionary",
					"source": "Bridge Connectivity Solutions Pvt. Ltd.",
					"license": "Creative Commons - Attribution-ShareAlike [CC BY-SA]",
					"duration_sec": duration,
					"total_frames": frame_count,
					"fps": fps,
					"width": width,
					"height":height,
					"transcript": {
						"text": text,
						"language": {
							"name": "English",
							"nameLocal" : "English",
							"ISO639-3": "eng",
							"BCP-47": "en"
						},
					}
					}
		with open(f"{output_path}/{id}.json", "w") as f:
			json.dump(metadata, f, indent=4)
		logging.info(f'Processed {id}!!!!!!!!!!!!!!!!!!!!')
	finally:
		clear_space(f"{id}_large.mp4")
		clear_space(f"{id}.mp4")
		clear_space(f"{id}_pose-mediapipe.pose")

def clear_space(file_path):
	if os.path.exists(file_path):
		os.remove(file_path)

def main():
	if len(sys.argv) < 4:
		print("Usage: python process_video_script.py <id> <input-video-path> <out-folder>")
		sys.exit(1)
	
	video_id = sys.argv[1]
	input_path = sys.argv[2]
	output_path = sys.argv[3]

	process_video(video_id, input_path, output_path)

if __name__ == '__main__':

	main()
