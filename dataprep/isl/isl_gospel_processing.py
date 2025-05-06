
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

junk_pattern_in_filename = re.compile(r'\-[\w]+')

def process_video(id, remote_path, nxt_cld_conn):
	nxt_cld_conn.download_video(remote_path, f"{id}_large.mp4")

	downsample_video_ondisk(f"{id}_large.mp4", f"{id}.mp4")

	trimmed_stream = trim_off_storyboard(None, id)
	try:
		if not trimmed_stream:
			raise Exception("Processing with mediapipe failed")
		generate_mask_and_pose_video_files(id)

		output_path = f"ISLGospels_processed/{id}.mp4"
		nxt_cld_conn.upload_file(f"{id}.mp4", output_path)
		nxt_cld_conn.upload_file(f"{id}_pose.mp4", f"ISLGospels_processed/{id}.pose.mp4")
		nxt_cld_conn.upload_file(f"{id}_mask.mp4", f"ISLGospels_processed/{id}.mask.mp4")

		parts = remote_path.split("/")
		ref = f"{parts[0]} {parts[1].replace("Ch ", "")} "
		verse = parts[-1].split(".")[0].split(" ")[-1]
		verse = re.sub(junk_pattern_in_filename, verse, "")
		ref = ref+verse

		metadata = {"filename": f"{id}.mp4",
					"pose":f"{id}.pose.mp4", 
					"mask":f"{id}.pose.mp4",
					"source": remote_path,
					"bible-ref": ref,
					"transcripts": [{
								"text": ref,
								"language": "English",
								"ISO 639-1": "en"
							}],
					"glosses": [{
								"text": [(0,0,"gloss")],
								"language": "English",
								"ISO 639-1": "en"

								}]
					}
		with open(f"{id}.json", "w") as f:
			json.dump(metadata, f, indent=4, sort_keys=True)
		nxt_cld_conn.upload_file(f"{id}.json", f"ISLGospels_processed/{id}.json")
		print(f'Uploaded {id}!!!!!!!!!!!!!!!!!!!!')
	except Exception as exce:
		print(exce)
		# raise exce
	finally:
		clear_space(f"{id}_large.mp4")
		clear_space(f"{id}.mp4")
		clear_space(f"{id}_mask.mp4")
		clear_space(f"{id}_pose.mp4")
		clear_space(f"{id}.json")
		pass

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
	if len(sys.argv) != 3:
		print("Usage: python process_video_script.py <path> <id>")
		sys.exit(1)

	video_id = int(sys.argv[1])
	video_path = sys.argv[2]

	# nxt_cld_conn = NextCloud_connection()
	output_path = "/mnt/share/processed_path"
	process_video_onmount(video_id, video_path, output_path)

def main_nxtcld():
	if len(sys.argv) != 3:
		print("Usage: python process_video_script.py <path> <id>")
		sys.exit(1)

	video_id = int(sys.argv[1])
	video_path = sys.argv[2]

	nxt_cld_conn = NextCloud_connection()
	process_video(video_id, video_path, nxt_cld_conn)

def list_videofile_inputs(path, count_start=0):
	nxt_cld_conn = NextCloud_connection()
	files = nxt_cld_conn.get_files(path)
	i = count_start
	for file in files:
		if not file.endswith("/"):
			i += 1
			print(f"{i}\t{path}{file}")
	return i

def list_videofile_inputs_onmount(path, count_start=0):
	files = [f for f in Path(path).iterdir() if f.is_file()]
	i = count_start
	for file in files:
		if not file.endswith("/"):
			i += 1
			print(f"{i}\t{path}{file}")
	return i

if __name__ == "__main__":
	# source_data_path = "/mnt/share"
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 1/", count_start=0)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 2/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 3/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 4/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 5/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 6/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 7/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 8/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 9/", count_start=num)
	# num = list_videofile_inputs_onmount(f"{source_data_path}/matthew/Ch 10/", count_start=num)
	
	# main()


	# num = list_videofile_inputs("/Matthew/Ch 1/", count_start=0)
	main_nxtcld()

