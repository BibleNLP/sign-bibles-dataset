import os
import sys
import ffmpeg

def convert_mp4_to_h264(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(folder, filename)
            temp_output = os.path.join(folder, 'temp_' + filename)
            print(f"Converting: {filename}")
            (
                ffmpeg
                .input(input_path)
                .output(
                    temp_output,
                    vcodec='libx264',
                    preset='fast',
                    crf=23,
                    acodec='copy'
                )
                .overwrite_output()
                .run()
            )
            os.replace(temp_output, input_path)
            print(f"Done: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python change_mp4_encoding.py <folder_path>")
        sys.exit(1)
    folder_path = sys.argv[1]
    convert_mp4_to_h264(folder_path)
