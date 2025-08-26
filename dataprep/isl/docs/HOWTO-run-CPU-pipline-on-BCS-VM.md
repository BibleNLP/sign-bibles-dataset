# The CPU Pipeline

All the processing that doesn't requires a GPU is done on a multi core CPU machine, [a VM at BCS](./HOWTO-connect-BCS-VM.md).

The key tools in this pipeline are:
* ffmpeg
* opencv python
* pose-format (mediapipe pose estimation)
* GNU parallel

Processing done:
* Down sample the input videos to lower resolution.
* Trim the start and end of the videos to remove clapper board and idle pose.
* Estimate frame wise pose using mediapipe and save output as `.pose` format.
* Add biblical metadata like verse reference range, list of BibleNLP vrefs and parallel Bible verse text(from BSB version).
* Also add more video specific info to metadata like number of frames, fps, width, height etc.


## Steps
### 1. Install ffmpeg

```bash
sudo apt-get install ffmpeg
```
### 2. Install GNU Parallel
```bash
sudo apt-get install parallel
```
### 3. Set up the code base, if not done already
```bash
git clone https://github.com/BibleNLP/sign-bibles-dataset.git
```
### 4. Install python dependencies
```bash
cd  sign-bibles-dataset/dataprep/isl
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-CPU-only.txt
```
### 5. Get list of videos in a book
Open the script `gospel_file_list.py` and edit the main method at the end of the file. Add the path to the folders where input data is. 

```bash
python gospel_file_list.py > gospel_list.txt
```

### 6. Process files parallely

```bash
bash run_parallel.sh
```
This command will take inputs from `gospel_list.txt` and run `n` jobs at a time to process the files using ffmpeg and mediapipe.

> To adjust the number of parallel jobs to be run, edit the bash script `run_parallel.sh`

The script will write logs to a `success.log`, a `fail.log` and an `app.log`. To change the location of these logs, edit `run_parallel.sh`.
