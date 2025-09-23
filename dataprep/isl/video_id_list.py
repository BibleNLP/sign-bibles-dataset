from pathlib import Path
import sys

# from nextcloud_connect import NextCloud_connection

# def list_videofile_inputs(path, count_start=0):
# 	nxt_cld_conn = NextCloud_connection()
# 	files = nxt_cld_conn.get_files(path)
# 	i = count_start
# 	for file in files:
# 		if not file.endswith("/"):
# 			i += 1
# 			print(f"{i}\t{path}{file}")
# 	return i

def list_videofile_inputs_onmount(path, count_start=0):
	files = [f for f in Path(path).rglob("*") if f.is_file()]
	i = count_start
	for file in files:
		if file.suffix.lower() == ".mp4":
			i += 1
			print(f"{i}\t{file}")
	return i

# def list_already_uploaded(path="/ISLGospels_processed/")
# 	nxt_cld_conn = NextCloud_connection()
# 	files = nxt_cld_conn.get_files(path)
# 	i = count_start
# 	for file in files:
# 		if file.endswith(".json"):
# 			metadata = nxt_cld_conn.download_video
# 			print(f"{i}\t{path}{file}")
# 	return i


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: python dataprep/isl/gospel_file_list.py <path_to_gospel_files> <count_start>")
		sys.exit(1)
	
	path = sys.argv[1]
	count_start = int(sys.argv[2])

	num = list_videofile_inputs_onmount(path, count_start=count_start)

	# num = list_videofile_inputs_onmount("/mnt/share/matthew/", count_start=0)
	# num = list_videofile_inputs_onmount("/mnt/share/mark/", count_start=num)
	# num = list_videofile_inputs_onmount("/mnt/share/luke/", count_start=num)
	# num = list_videofile_inputs_onmount("/mnt/share/john/", count_start=num)

	# num = list_videofile_inputs_onmount("/mnt/share/original/", count_start=0)