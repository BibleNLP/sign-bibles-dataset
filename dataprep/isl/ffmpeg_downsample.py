import io
import ffmpeg


def downsample_video(video_stream):
    output_stream = io.BytesIO()
    process = (
        ffmpeg
        .input('pipe:', format='mp4')  # Ensure input format is correct
        .output('pipe:', format='mpegts', vf='scale=640:-2', vcodec='libx264', preset='ultrafast', crf=28)
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
    