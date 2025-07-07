from pathlib import Path

from nextcloud_connect import NextCloud_connection


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


# def list_already_uploaded(path="/ISLGospels_processed/")
# 	nxt_cld_conn = NextCloud_connection()
# 	files = nxt_cld_conn.get_files(path)
# 	i = count_start
# 	for file in files:
# 		if file.endswith(".json"):
# 			metadata = nxt_cld_conn.download_video
# 			print(f"{i}\t{path}{file}")
# 	return i


if __name__ == "__main__":
    # num = list_videofile_inputs("/Matthew/Ch 1/", count_start=0)

    # num = list_videofile_inputs("/Matthew/Ch 2/", count_start=12)
    # num = list_videofile_inputs("/Matthew/Ch 3/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 4/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 5/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 6/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 7/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 8/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 9/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 10/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 11/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 12/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 13/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 14/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 15/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 16/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 17/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 18/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 19/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 20/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 21/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 22/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 23/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 24/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 25/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 26/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 27/", count_start=num)
    # num = list_videofile_inputs("/Matthew/Ch 28/", count_start=num)

    num = list_videofile_inputs("/Mark/Ch 1/", count_start=243)
    num = list_videofile_inputs("/Mark/Ch 2/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 3/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 4/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 5/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 6/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 7/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 8/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 9/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 10/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 11/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 12/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 13/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 14/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 15/", count_start=num)
    num = list_videofile_inputs("/Mark/Ch 16/", count_start=num)
