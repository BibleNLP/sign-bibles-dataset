import io
import os
import sys
import ffmpeg


def downsample_video(video_stream):
    output_stream = io.BytesIO()
    process = (
        ffmpeg
        .input('pipe:', format='mp4')  # Ensure input format is correct
        .output('pipe:', format='mpegts', vf='scale=-2:1080', vcodec='libx264', preset='ultrafast', crf=28)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, overwrite_output=True)
    )
    out, _ = process.communicate(input=video_stream.read())

    output_stream.write(out)
    output_stream.seek(0)

    size = output_stream.getbuffer().nbytes
    if size > 0:
        # print(f"Downsampling successful! File size: {size} bytes")
        pass
    else:
        raise ValueError(f"Downsampling failed! File is empty.")
    return output_stream


def downsample_video_ondisk(input_path, output_path):
    """
    Downsamples a video from disk and writes the result as MP4.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the downsampled video (should end with .mp4).
    """
    try:
        process = (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                format='mp4',                   # output as MP4
                vf='scale=-2:1080',              # resize height to 1080, preserve aspect ratio
                vcodec='libx264',
                preset='ultrafast',
                crf=28
            )
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")

        if os.path.getsize(output_path) == 0:
            raise ValueError("Downsampling failed! Output file is empty.")

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        raise

def main(input_video_path, output_video_path):
    """
    Recursively downsample all .mp4/.MP4 files in input_video_path,
    preserving the directory structure in output_video_path.
    """
    for root, dirs, files in os.walk(input_video_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                input_path = os.path.join(root, file)
                # Compute relative path from input_video_path
                rel_path = os.path.relpath(input_path, input_video_path)
                output_path = os.path.join(output_video_path, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(f"Downsampling {input_path} to {output_path}")
                downsample_video_ondisk(input_path, output_path)


if __name__ == "__main__":
    if sys.argv[1:]:
        input_video_dir = sys.argv[1]
        output_video_dir = sys.argv[2]
    else:
        input_video_dir = "/home/kavitha/Documents/Sign/ISL dataset/test_downsampling_script/raw/"
        output_video_dir = "/home/kavitha/Documents/Sign/ISL dataset/test_downsampling_script/downsampled/"
    os.makedirs(output_video_dir, exist_ok=True)

    main(input_video_dir, output_video_dir)

