# Rank Segments In a Video Task

## Description
Given segments in a video and a verse, rank the segments according to relevance.

Inputs: 
* The text of a verse
* segment indices in a video as well as their start and end frames 

Example Input: "It is a silly place" has two segments in the video which are 'relevant' and should be retrieved.
```
seg_idx,query_text,video_id,start_frame,end_frame,total_frames
0,It is a silly place.,video_01,0,100,10000
1,The Castle Aaargh!,video_01,100,200,10000
2,It is a silly place.,video_01,300,400,10000
3,This is an ex-parrot!,video_01,400,500,10000
````
Output:
* rank every segment for every query, lower rank is more relevant

Example correct predictions for query "It is a silly place":
```
video_id,query_text,rank,seg_idx
video_01,It is a silly place.,0,0 # segment 0 ranked highly, good!
video_01,It is a silly place.,1,2 # segment 2 also ranked highly, great!
video_01,It is a silly place.,1,1 # Not relevant for this query, placed below the others
video_01,It is a silly place.,1,3 # Also not relevant for this query, placed below the others
```

## Evaluation
Either put results in a CSV and use 

```
python sign_bibles_dataset/tasks/text2sign/extract_ground_truth_from_webdataset.py --language-subset ase
python sign_bibles_dataset/tasks/text2sign/rank_segments_in_video/evaluate_predictions.py path/to/ground_truth.csv sign-bibles-dataset/sign_bibles_dataset/tasks/text2sign/rank_segments_in_video/example_predictions/wrong_predictions.csv 
```