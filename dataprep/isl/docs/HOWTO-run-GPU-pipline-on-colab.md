# GPU Pipeline for DWPose

**Input**: The CPU processed video files named as `<id>.mp4`.

**Output**: `<id>.npz` files that have frame wise pose coordinates and confidence values for the input sign videos. 

**Models**: The two pre-trained models used for dwpose are to be obtained:
```bash
gdown --id 12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2 -O models/dw-ll_ucoco_384.onnx
gdown --id 1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI -O models/yolox_l.onnx
```


[This google colab](./) notebook that has the steps and code for running the pipeline. The major steps involved in it are:

1. Getting the input data and models to the GPU instance, in this case colab.
2. Setting up code base.
3. Preparing the input id list in `input_list.txt`.
4. Running `n` parallel jobs via `run_parallel_gpu.sh`. The number of job can be set based on the available GPU RAM. For a 40 GB A100 GPU, `n` can be 25.
5. Moving the output data to a persistant storage.

> As google colab sessions can be lost mid run and the data generated can be all lost, it is recommended that steps 3-5 are run repeatedly for smaller batches, say 100 videos.


> The paths for log files are set in `run_parallel_gpu.sh` and start of `dwpose_processing.py`.

> If there is a need to change the model path as set in colab, edit it in `dwpose/wholebody.py`

> If the data paths are not as per the preset values, edit them in `generate_pose_files_v2()` method in `dwpose_processing.py`.