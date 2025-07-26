The idea is to emulate Studying and Mitigating Biases in Sign Language Understanding Models

# pipeline

```
conda create -n sign-bible-analysis python=3.12
pip install deepface skin-tone-classifier pyarrow pandas brisque tf-keras
sign-bible-analysis
```


1. Extract frames to videostem_frames folders. E.g. foo.mp4 -> foo_frames.mp4
```
# extract with 100 workers, default 1 frame every 10 seconds
./sign_bibles_dataset/data_analysis/0_extract_frames.sh samples/ 100
# 1 frame every second based on framerate.
#./sign_bibles_dataset/data_analysis/extract_frames.sh samples/ 100 1s 
# every 100 frames
#./sign_bibles_dataset/data_analysis/extract_frames.sh samples/ 100 100 
```
2. Calculate and save Brisque scores to the frames folders with https://pypi.org/project/brisque/

```
./sign_bibles_dataset/data_analysis/1_run_brisque_scores.sh samples/
```

2. Analyze faces with DeepFace
I haven't been able to get this working usefully. In "God Creates the World" it finds "45 unique faces" in 45 frames. 
