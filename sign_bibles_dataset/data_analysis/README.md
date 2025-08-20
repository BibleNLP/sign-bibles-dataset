The idea is to emulate [Studying and Mitigating Biases in Sign Language Understanding Models](https://arxiv.org/abs/2410.05206)


# Stats: 

Per project/language... 
* Demographics of Signers: Unique Signers, Genders, Races
* Video length distributions


## Video lengths

```
dir,total_duration_sec,duration_formatted
total,2.49136e+06,692:02:40.00
tza,210722,58:32:02.00
eth,151375,42:02:55.00
rsl,93973.3,26:06:13.30
lsb,175676,48:47:56.00
ase,169671,47:07:51.00
mis,670573,186:16:13.00
nsp,12772.1,03:32:52.10
bqn,94365.1,26:12:45.10
eso,9611.32,02:40:11.32
ins,22665.3,06:17:45.30
mzy,76951.3,21:22:31.30
esl,9498.44,02:38:18.44
nsi,187088,51:58:08.00
ugn,198895,55:14:55.00
xki,260339,72:18:59.00
sqs,3617.55,01:00:17.55
gse,143563,39:52:43.00
```

# Processing Pipeline

```
conda create -n sign-bible-analysis python=3.12
pip install deepface skin-tone-classifier pyarrow pandas brisque tf-keras
sign-bible-analysis
```


1. Extract 1 frame every 10 secondss to videostem_frames folders. E.g. foo.mp4 -> foo_frames.mp4
```
# extract with 100 workers, default 1 frame every 10 seconds
./sign_bibles_dataset/data_analysis/0_extract_frames.sh samples/ 100
# Second arg: 1 frame every second based on framerate.
#./sign_bibles_dataset/data_analysis/extract_frames.sh samples/ 100 1s 
# ... or every 100 frames
#./sign_bibles_dataset/data_analysis/extract_frames.sh samples/ 100 100 
```
2. Calculate and save Brisque scores to the frames folders with https://pypi.org/project/brisque/

```
# use 100 workers to calculate brisque scores
./sign_bibles_dataset/data_analysis/1_run_brisque_scores.sh samples/ 100
```

2. Analyze faces with DeepFace
I haven't been able to get this working usefully to count unique signers. 
In "God Creates the World" it finds "45 unique faces" in 45 frames. 

