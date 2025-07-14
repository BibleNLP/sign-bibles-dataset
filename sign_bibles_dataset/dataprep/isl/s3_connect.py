import io
import os

import boto3
import botocore
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class S3_connection:
    def __init__(
        self,
        bucket_name=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        if aws_access_key_id is None:
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        if aws_secret_access_key is None:
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if bucket_name is None:
            bucket_name = os.getenv("BUCKET_NAME")
        self.bucket_name = bucket_name

        cfg = botocore.client.Config(max_pool_connections=os.cpu_count() * 5)

        # Create an S3 client
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=cfg,
        )

    def download_video(self, bucket_name, input_key):
        if bucket_name is None:
            bucket_name = self.bucket_name
        video_stream = io.BytesIO()
        self.s3.download_fileobj(bucket_name, input_key, video_stream)
        video_stream.seek(0)

        size = video_stream.getbuffer().nbytes  # or len(video_stream.getvalue())
        if size > 0:
            # print(f"Download successful! File size: {size} bytes")
            pass
        else:
            raise ValueError("Download failed! File is empty.")
        return video_stream

    def upload_file(self, bucket_name, video_stream, output_key):
        if bucket_name is None:
            bucket_name = self.bucket_name
        self.s3.upload_fileobj(video_stream, bucket_name, output_key)
        # print(f"Uploaded to {bucket_name}/{output_key}")

    def get_files(self, bucket_name=None):
        if bucket_name is None:
            bucket_name = self.bucket_name
        paginator = self.s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name)

        all_objects = []
        for page in page_iterator:
            if "Contents" in page:
                all_objects.extend(page["Contents"])

        # Print total count and file names
        print(f"Total number of objects in bucket '{bucket_name}': {len(all_objects)}")
        original_files = [
            obj["Key"]
            for obj in all_objects
            if obj["Key"].startswith("Original/") and (obj["Key"].endswith(".MP4") or obj["Key"].endswith(".mp4"))
        ]
        print(f"Total number of videos in Original: {len(original_files)}")
        return original_files
