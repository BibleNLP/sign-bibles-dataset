# ISL data accessing and processing 

ISL bible and dictionary data are made available by BCS. They are originally published on Youtube and distributed with a cc-by-sa licence.

## Setup and run

Follow the below installations and configurations.

### Install ffmpeg

```bash
sudo apt-get install ffmpeg
```

### Setup DWPose

```bash
git clone https://github.com/IDEA-Research/DWPose.git
wget -O DWPose/ControlNet-v1-1-nightly/models/control_v11p_sd15_openpose.pth https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_openpose.pth
wget -O DWPose/ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt
gdown --id 12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2 -O DWPose/ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx
gdown --id 1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI -O DWPose/ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx
```

The full path to the annotator folder should be updated in the file `DWPose/ControlNet-v1-1-nightly/annotator/dwpose/wholebody.py`. 

The path to where `DWPose/ControlNet-v1-1-nightly` folder is added to `.env` file as `DWPOSE_PATH`.


### Connection to S3 bucket
Set the following environment variables in the `.env` file.

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
BUCKET_NAME
```

### Install python dependencies
Good to install these in a virtual environment.

```bash
python -m venv ENV 
source ENV/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```
