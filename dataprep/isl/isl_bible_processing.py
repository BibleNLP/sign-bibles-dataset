
import os
import re
import asyncio
import json
import sys
import shutil
import logging
from pathlib import Path
import ffmpeg

# from mediapipe_trim import trim_off_storyboard
# from pose_format_util import video2poseformat
from bible_text_access import get_verses, book_code_lookup
from biblenlp_util import ref2vref

logging.basicConfig(filename='/content/logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

verse_john_pattern = re.compile(r"(\d+(\-\d+)?)\.MP4")
verse_generic_pattern = re.compile(r"(\d+(\-\d+)?)")

def process_video(id, input_path, output_path):
	if output_path.endswith("/"):
		output_path = output_path[:-1]
	main_path = f"{output_path}/{id}.mp4"
	try:
		# shutil.copy(input_path, f"./{id}.mp4")  


		# trimmed_stream = trim_off_storyboard(None, id)
		# if not trimmed_stream:
		# 	raise Exception("Processing with mediapipe failed")
		# video2poseformat(id) #  .pose format using mediapipe

		# shutil.move(f"{id}.mp4", output_path)
		# shutil.move(f"{id}_pose-mediapipe.pose", f"{output_path}/{id}.pose-mediapipe.pose")

		parts = input_path.split("/")

		signer = "Signer_1"
		if parts[3]=="john":
			ref = f"{book_code_lookup[parts[3]]} {parts[4].replace("Ch-", "")}"
			ver_match = re.search(verse_john_pattern, parts[-1])
			if not ver_match:
				raise Exception(f"Cant compose reference from:{parts}")
			verse_parts = ver_match.group(1)
			signer = "Signer_2"
		elif parts[3] in ["matthew", "mark", "luke"]:
			ref = f"{book_code_lookup[parts[3]]} {parts[4].replace("Ch ", "")}"
			verse = parts[-1].split(".")[0].split(" ")[1]
			verse_parts = "-".join(verse.split("-")[:-1])
		else:
			book = parts[3]
			chapter = re.search(r"(\d+)", parts[4]).group(0)
			verse = re.search(verse_generic_pattern, parts[-1]).group(0)
			ref = f"{book_code_lookup[book]} {chapter}"
			verse_parts = verse

		ref = f"{ref}:{verse_parts}"
		vref = ref2vref(ref)

		probe = ffmpeg.probe(input_path) # probe = ffmpeg.probe(main_path)
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
					"project": "Indian Sign Language Bible (ISLV)",
					"source": "https://www.youtube.com/@islv-holybible",
					"license": "Creative Commons - Attribution-ShareAlike [CC BY-SA]",
					"bible-ref": ref,
					"biblenlp-vref": vref,
					"duration_sec": duration,
					"signer": signer,
					"total_frames": frame_count,
					"fps": fps,
					"width": width,
					"height":height,
					}
		transcripts = [{
					"text": get_verses(ref, "BSB"),
					"start_frame": 0,
					"end_frame": frame_count,
					"language": {
						"name": "English",
						"nameLocal" : "English",
						"ISO639-3": "eng",
						"BCP-47": "en-US"
					},
					"license": "public domain",
					"source": "http://ebible.org/engbsb/",
					"bible-ref": ref,
					"biblenlp-vref": vref,
				}]
		with open(f"{output_path}/{id}.json", "w") as f:
			json.dump(metadata, f, indent=4)
		with open(f"{output_path}/{id}.transcripts.json", "w") as f:
			json.dump(transcripts, f, indent=4)
		logging.info(f'Processed {id}!!!!!!!!!!!!!!!!!!!!')
	finally:
		clear_space(f"{id}_large.mp4")
		clear_space(f"{id}.mp4")
		clear_space(f"{id}_pose-mediapipe.pose")

def clear_space(file_path):
	if os.path.exists(file_path):
		os.remove(file_path)


def main():
	if len(sys.argv) != 4:
		print("Usage: python process_video_script.py <id> <input-video-path> <out-folder>")
		sys.exit(1)

	video_id = int(sys.argv[1])
	video_path = sys.argv[2]
	output_path = sys.argv[3]

	process_video(video_id, video_path, output_path)


if __name__ == "__main__":

	# process_video("2", "/Matthew/Ch 1/10 17-0D5A1069.MP4", "../../../Matthew_processed")

	main()


