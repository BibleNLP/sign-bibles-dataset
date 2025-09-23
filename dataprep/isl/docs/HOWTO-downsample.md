# FFMPEG downsampling

### 1. Install ffmpeg

```bash
sudo apt-get install ffmpeg
```
### 2. Set up the code base, if not done already
```bash
git clone https://github.com/BibleNLP/sign-bibles-dataset.git
```
### 3. Install python dependencies
```bash
cd  sign-bibles-dataset/dataprep/isl
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local-ffmpeg.txt
```

### 4. Run
Input directory should have all the HD files. This folder can have sub folders. Only the `.mp4` files will be considered as input from there.

```bash
python ffmpeg_downsample.py <input_dir> <output_dir>
```
Output directory will be created if not present. preserving the directory structure and file names of input directory.
