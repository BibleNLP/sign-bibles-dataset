
import os
import re
import asyncio
import json
import sys
import shutil
from pathlib import Path

from nextcloud_connect import NextCloud_connection
from ffmpeg_downsample import downsample_video_ondisk
from mediapipe_trim import trim_off_storyboard
from dwpose_processing import generate_mask_and_pose_video_files
from bible_text_access import get_verses

def process_video(id, remote_path, nxt_cld_conn, output_path):
	if output_path.endswith("/"):
		output_path = output_path[:-1]
	try:
		nxt_cld_conn.download_file(remote_path, f"{id}_large.mp4")

		downsample_video_ondisk(f"{id}_large.mp4", f"{id}.mp4")

		trimmed_stream = trim_off_storyboard(None, id)
		if not trimmed_stream:
			raise Exception("Processing with mediapipe failed")
		generate_mask_and_pose_video_files(id)

		main_path = f"{output_path}/{id}.mp4"
		shutil.move(f"{id}.mp4", output_path)
		shutil.move(f"{id}_pose.mp4", f"{output_path}/{id}.pose.mp4")
		shutil.move(f"{id}_mask.mp4", f"{output_path}/{id}.mask.mp4")

		parts = remote_path.split("/")
		ref = f"{parts[1]} {parts[2].replace("Ch ", "")}"
		verse = parts[-1].split(".")[0].split(" ")[1]
		verse_parts = "-".join(verse.split("-")[:-1])
		ref = f"{ref}:{verse_parts}"

		metadata = {"filename": f"{id}.mp4",
					"pose":f"{id}.pose.mp4", 
					"mask":f"{id}.pose.mp4",
					"source": f"{ref} of https://www.youtube.com/@islv-holybible",
					"license": "CC-BY-SA",
					"bible-ref": ref,
					"transcripts": [{
								"text": get_verses(ref),
								"language": "English",
								"ISO 639-1": "en"
							}],
					"glosses": [{
								"text": [(0,0,"nil")],
								"language": "English",
								"ISO 639-1": "en"

								}]
					}
		# print(f"{metadata['bible-ref']} ---> {metadata['transcripts'][0]['text']}")
		with open(f"{id}.json", "w") as f:
			json.dump(metadata, f, indent=4)
		shutil.move(f"{id}.json", f"{output_path}/{id}.json")
		print(f'Processed {id}!!!!!!!!!!!!!!!!!!!!')
	finally:
		clear_space(f"{id}_large.mp4")
		clear_space(f"{id}.mp4")
		clear_space(f"{id}_mask.mp4")
		clear_space(f"{id}_pose.mp4")
		clear_space(f"{id}.json")

def clear_space(file_path):
	if os.path.exists(file_path):
		os.remove(file_path)


def process_video_onmount(id, orig_path, processed_path):

	downsample_video_ondisk(orig_path, f"{id}.mp4")

	trimmed_stream = trim_off_storyboard(None, id)
	try:
		if not trimmed_stream:
			raise Exception("Processing with mediapipe failed")
		generate_mask_and_pose_video_files(id)

		output_path = f"{processed_path}/{id}.mp4"
		shutil.move(f"{id}.mp4", output_path)
		shutil.move(f"{id}_pose.mp4", f"{output_path}/{id}.pose.mp4")
		shutil.move(f"{id}_mask.mp4", f"{output_path}/{id}.mask.mp4")

		parts = orig_path.split("/")
		ref = f"{parts[-3]} {parts[-2].replace("Ch ", "")}"
		verse = parts[-1].split(".")[0].split(" ")[1]
		verse_parts = "-".join(verse.split("-")[:-1])
		ref = f"{ref}:{verse_parts}"


		parts = remote_path.split("/")
		ref = f"{parts[0]} {parts[1].replace("Ch ", "")} "
		verse = parts[-1].split(".")[0].split(" ")[-1]
		verse = re.sub(junk_pattern_in_filename, verse, "")
		ref = ref+verse

		metadata = {"filename": f"{id}.mp4",
					"pose":f"{id}.pose.mp4", 
					"mask":f"{id}.pose.mp4",
					"source": f"{ref} of https://www.youtube.com/@islv-holybible",
					"license": "CC-BY-SA",
					"bible-ref": ref,
					"transcripts": [{
								"text": get_verses(ref),
								"language": "English",
								"ISO 639-1": "en"
							}],
					"glosses": [{
								"text": [(0,0,"nil")],
								"language": "English",
								"ISO 639-1": "en"

								}]
					}
		with open(f"{id}.json", "w") as f:
			json.dump(metadata, f, indent=4, sort_keys=True)
		shutil.move(f"{id}.json", f"{output_path}/{id}.json")
		print(f'processed {id}!!!!!!!!!!!!!!!!!!!!')
	except Exception as exce:
		# print(exce)
		raise exce



def main():
	if len(sys.argv) != 4:
		print("Usage: python process_video_script.py <path> <id>")
		sys.exit(1)

	video_id = int(sys.argv[1])
	video_path = sys.argv[2]

	# nxt_cld_conn = NextCloud_connection()
	output_path = "/mnt/share/processed_path"
	process_video_onmount(video_id, video_path, output_path)

def main_nxtcld():
	if len(sys.argv) != 4:
		print("Usage: python process_video_script.py <path> <id>")
		sys.exit(1)

	video_id = int(sys.argv[1])
	video_path = sys.argv[2]
	output_path = sys.argv[3]

	nxt_cld_conn = NextCloud_connection()
	process_video(video_id, video_path, nxt_cld_conn, output_path)


if __name__ == "__main__":

	# process_video("244", "/Mark/Ch 1/1 1-3-A082C001_221013_H10P (New).MP4", NextCloud_connection(), "../../../Mark_processed")
	# main()

	main_nxtcld()


