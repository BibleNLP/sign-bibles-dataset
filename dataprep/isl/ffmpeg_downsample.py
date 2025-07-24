import io
import os
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