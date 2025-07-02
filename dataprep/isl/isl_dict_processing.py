import json
import os
import io
import glob
import sys
from pathlib import Path
import shutil
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import ffmpeg
import datetime
import logging

# from s3_connect import S3_connection
from ffmpeg_downsample import downsample_video
from pose_format_util import video2poseformat
from mediapipe_trim import trim_off_storyboard
from dwpose_processing import generate_pose_files_v2


# Load environment variables from .env
load_dotenv()
bucket_name = os.getenv("BUCKET_NAME")

logging.basicConfig(filename='logs/app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_video_v2(id, input_path, output_path):
	"""Pipeline:
		--> use already downsampled and downloaded videos
		--> process: pose estimation, metadata etc
		--> keep in a local path for tar building later"""
	main_path = Path(output_path) / f"{id}.mp4"
	try:
		# video_stream = s3_conn.download_video(bucket_name, input_key)

		# downsample_video_ondisk(f"{id}_large.mp4", f"{id}.mp4")
		shutil.copy(input_path, f"./{id}.mp4") 
		

		trimmed_stream = trim_off_storyboard(None, id)
		if not trimmed_stream:
			raise Exception("Processing with mediapipe failed")
		video2poseformat(id) #  .pose format using mediapipe
		generate_pose_files_v2(id) # mp4, and npz usging dwpose

		shutil.move(f"{id}.mp4", main_path)
		shutil.move(f"{id}_pose-animation.mp4", f"{output_path}/{id}.pose-animation.mp4")
		shutil.move(f"{id}_pose-mediapipe.pose", f"{output_path}/{id}.pose-mediapipe.pose")
		shutil.move(f"{id}_pose-dwpose.npz", f"{output_path}/{id}.pose-dwpose.npz")

		text = read_old_json(input_path.replace(".mp4", ".json"))

		probe = ffmpeg.probe(main_path)
		duration = float(probe['format']['duration'])

		signer = "xxx"

		metadata = {"filename": f"{id}.mp4",
					"pose": {
						"animation": f"{id}.pose-animation.mp4",
						"mediapipe": f"{id}.pose-mediapipe.pose",
						"dwpose": f"{id}.pose-dwpose.npz"
					},
					# "source": "https://www.youtube.com/@islv-holybible",
					"license": "CC-BY-SA",
					"language": {
						"name": "Indian Sign Language",
						"ISO639-3": "ins",
						"BCP-47": "ins-IN"
					},
					# "bible-ref": ref,
					# "biblenlp-vref": vref,
					"duration": f"{duration} seconds",
					"signer": signer,
					"transcripts": [{
								"text": text,
								"language": {
									"name": "English",
									"ISO639-3": "eng",
									"BCP-47": "en-US"
								},
								"source": "Berean Standard Bible",
							}],
					"glosses": [{
								"text": [(0,duration,text)],
								"language": {
									"name": "English",
									"ISO639-3": "eng",
									"BCP-47": "en"
								}

								}]
					}
		with open(f"{output_path}/{id}.json", "w") as f:
			json.dump(metadata, f, indent=4)
		print(f'Processed {id}!!!!!!!!!!!!!!!!!!!!')
	finally:
		clear_space(f"{id}_large.mp4")
		clear_space(f"{id}.mp4")
		clear_space(f"{id}_pose-animation.mp4")
		clear_space(f"{id}_pose-dwpose.npz")
		clear_space(f"{id}_pose-mediapipe.pose")

def clear_space(file_path):
	if os.path.exists(file_path):
		os.remove(file_path)

def read_old_json(file_name):
	metadata = {}
	with open(file_name, 'r', encoding="utf-8") as fp:
		metadata = json.load(fp)
	return metadata['text-transcript']

def process_video(input_key, id, s3_conn):
	"""Full pipeline: download -> downsample -> upload"""
	video_stream = s3_conn.download_video(bucket_name, input_key)
	downsampled_stream = downsample_video(video_stream)

	temp_file1 = open(f"{id}.mp4", 'w+b')
	temp_file1.write(downsampled_stream.getvalue())  # Write video data
	temp_file1.close()  # Close so other processes can access it
	
	trimmed_stream = trim_off_storyboard(downsampled_stream, id)
	try:
		if not trimmed_stream:
			raise Exception("Processing with mediapipe failed")
	  	# clear_output(wait=True)

		mask_stream, pose_stream = generate_mask_and_pose_video_streams(id)
		
		file_name = input_key.split("/")[-1].split(".")[0]
		output_key = f"For-ISL-dataset2/{id}.mp4"
		s3_conn.upload_file(bucket_name, trimmed_stream, output_key)
		s3_conn.upload_file(bucket_name, pose_stream, f"For-ISL-dataset2/{id}.pose.mp4")
		s3_conn.upload_file(bucket_name, mask_stream, f"For-ISL-dataset2/{id}.mask.mp4")

		metadata = {"filename": f"{file_name}.mp4", "text-transcript": file_name.lower().replace("-",""), 
			  "pose":f"{id}.pose.mp4", "mask":f"{id}.pose.mp4"}
		s3_conn.upload_file(bucket_name, io.BytesIO(json.dumps(metadata).encode('utf-8')), f"For-ISL-dataset2/{id}.json")
		print(f'Uploaded {id}!!!!!!!!!!!!!!!!!!!!')
	except Exception as exce:
		print(exce)
	finally:
		os.remove(temp_file1.name)

def get_cloud_files(s3_conn):
	original_files = s3_conn.get_files(bucket_name)
	return original_files


def main():
	s3_conn = S3_connection()

	original_files = get_cloud_files(s3_conn)


	# # Run with ThreadPoolExecutor
	# with ThreadPoolExecutor(max_workers=10) as executor:
	#     executor.map(process_video, original_files[1296:], range(1296, len(original_files), s3_conn))

	for i in range(0,10):
		file_key = original_files[i]
		# if i < 650:
		#   continue
		print(i)
		process_video(file_key, i, s3_conn)

def main_v2():
	if len(sys.argv) != 4:
		print("Usage: python process_video_script.py <path> <id>")
		sys.exit(1)

	video_id = int(sys.argv[1])
	video_path = sys.argv[2]
	output_path = sys.argv[3]

	process_video_v2(video_id, video_path, output_path)

def main_seqential(input_list, output_path):
	inp_args = []
	with open(input_list, 'r', encoding='utf-8') as fp:
		inp_lines = fp.readlines()
		inp_args = [line.split('\t')
						for line in inp_lines]

	now = datetime.datetime.now()
	success_log = open("logs/success.log", "w", encoding='utf-8')
	success_log.write(f"-------------start{now:%Y-%m-%d %H:%M:%S}----------\n")
	success_log.close()

	fail_log = open("logs/fail.log", "w", encoding='utf-8')
	fail_log.write(f"-------------start{now:%Y-%m-%d %H:%M:%S}----------\n")
	fail_log.close()

	for args in inp_args:
		try:
			process_video_v2(args[0].strip(), args[1].strip(), output_path)
			with open("logs/success.log", "a", encoding='utf-8') as success_log:
				success_log.write(f"{args[0]}\t{args[1]}")
		except Exception as exce:
			logging.exception(f"{id} Errored out!!!")
			with open("logs/fail.log", "a", encoding='utf-8') as fail_log:
				fail_log.write(f"{args[0]}\t{args[1]}")

if __name__ == '__main__':
	# main()

	# main_v2()

	main_seqential("dict_retry.txt", "../../../Dictionary_processed2/")
