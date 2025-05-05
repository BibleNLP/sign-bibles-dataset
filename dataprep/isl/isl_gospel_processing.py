
import os
import re
import asyncio
import json

from nextcloud_connect import NextCloud_connection
from ffmpeg_downsample import downsample_video_ondisk
from mediapipe_trim import trim_off_storyboard
from dwpose_processing import generate_mask_and_pose_video_files

junk_pattern_in_filename = re.compile(r'\-[\w]+')

def process_video(remote_path, id, nxt_cld_conn):
	# nxt_cld_conn.download_video(remote_path, f"{id}_large.mp4")

	# downsample_video_ondisk(f"{id}_large.mp4", f"{id}.mp4")

	# trimmed_stream = trim_off_storyboard(None, id)
	try:
	# 	if not trimmed_stream:
	# 		raise Exception("Processing with mediapipe failed")
	# 	generate_mask_and_pose_video_files(id)

	# 	output_path = f"ISLGospels_processed/{id}.mp4"
	# 	nxt_cld_conn.upload_file(f"{id}.mp4", output_path)
	# 	nxt_cld_conn.upload_file(f"{id}_pose.mp4", f"ISLGospels_processed/{id}.pose.mp4")
	# 	nxt_cld_conn.upload_file(f"{id}_mask.mp4", f"ISLGospels_processed/{id}.mask.mp4")

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
	finally:
		# os.remove(f"{id}_large.mp4")
		# os.remove(f"{id}.mp4")
		# os.remove(f"{id}_mask.mp4")
		# os.remove(f"{id}_pose.mp4")
		# os.remove(f"{id}.json")
		pass

def main():
	nxt_cld_conn = NextCloud_connection()

	print(nxt_cld_conn.get_files("/Matthew/"))

	process_video("/Matthew/Ch 1/1 1-0D5A1049.MP4", 1, nxt_cld_conn)

if __name__ == '__main__':
	# asyncio.run(main())
	main()
