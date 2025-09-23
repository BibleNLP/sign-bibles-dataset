# The CPU Pipeline

All the processing that doesn't requires a GPU is done on a multi core CPU instance , on Colab Pro.

The key tools in this pipeline are:
* ffmpeg
* opencv python
* pose-format (mediapipe pose estimation)
* GNU parallel

Processing done:
* Trim the start and end of the videos to remove clapper board and idle pose.
* Estimate frame wise pose using mediapipe and save output as `.pose` format.
* Add biblical metadata like verse reference range, list of BibleNLP vrefs and parallel Bible verse text(from BSB version).
* Also add more video specific info to metadata like number of frames, fps, width, height etc.


# GPU Pipeline for DWPose

**Input**: SD  `mp4` videos, with folders and file names indicating bible references.

**Output**: `<id>.json` , `<id>.pose-mediapipe.pose` and `<id>.transcripts.json` files for each input video.


[This google colab](./ISL_Dataset_CPU_pipline.ipynb) notebook has the steps and code for running the pipeline. The major steps involved in it are:

1. Getting the input data to the colab instance.
2. Setting up code base.
3. Preparing the input id list in `input_list.txt`.
4. Running `n` parallel jobs via `run_parallel_cpu.sh`. The number of job can be set based on the available GPU RAM. For a `v6e-1 TPU` instance, `n` can be 40.
5. Moving the output data to a persistant storage.

> As google colab sessions can be lost mid run and the data generated can be all lost, it is recommended that steps 3-5 are run repeatedly for smaller batches, say 100 videos.

The script will write logs to a `success.log`, a `fail.log` and an `app.log` within logs folder. To change the location of these logs, edit `run_parallel_cpu.sh`.

> Use `nproc` to find the number of CPU cores available and the number of parallel jobs accordingly.
