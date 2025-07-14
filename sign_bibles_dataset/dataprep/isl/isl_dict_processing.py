import json
import os
import io
from dotenv import load_dotenv

from s3_connect import S3_connection
from ffmpeg_downsample import downsample_video
from mediapipe_trim import trim_off_storyboard
from dwpose_processing import generate_mask_and_pose_video_streams


# Load environment variables from .env
load_dotenv()
bucket_name = os.getenv("BUCKET_NAME")


def process_video(input_key, id, s3_conn):
    """Full pipeline: download -> downsample -> upload"""
    video_stream = s3_conn.download_video(bucket_name, input_key)
    downsampled_stream = downsample_video(video_stream)

    temp_file1 = open(f"{id}.mp4", "w+b")
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

        metadata = {
            "filename": f"{file_name}.mp4",
            "text-transcript": file_name.lower().replace("-", ""),
            "pose": f"{id}.pose.mp4",
            "mask": f"{id}.pose.mp4",
        }
        s3_conn.upload_file(
            bucket_name,
            io.BytesIO(json.dumps(metadata).encode("utf-8")),
            f"For-ISL-dataset2/{id}.json",
        )
        print(f"Uploaded {id}!!!!!!!!!!!!!!!!!!!!")
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

    for i in range(0, 10):
        file_key = original_files[i]
        # if i < 650:
        #   continue
        print(i)
        process_video(file_key, i, s3_conn)


if __name__ == "__main__":
    main()
