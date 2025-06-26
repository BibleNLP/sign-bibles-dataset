import subprocess

def video2poseformat(id, model="mediapipe"):
	result = subprocess.run([
	    'video_to_pose',
	    '--format', model,
	    '-i', f'{id}.mp4',
	    '-o', f'{id}_pose-mediapipe.pose'
	], capture_output=True, text=True)

	# Check if the command was successful
	if result.returncode != 0:
		raise Exception(f"Error in generating pose format:{result.stderr}")