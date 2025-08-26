# ISL data accessing and processing 

ISL bible and dictionary data are made available by BCS. They are originally published on Youtube and distributed with a cc-by-sa licence.

## The dataprocessing workflow 


**Input**: Raw HD videos of Sign Bibles and Dictionary from BCS studios.

**Processing**:
1. Down sample the input videos to lower resolution.Also trim the start and end of the videos to remove clapper board and idle pose.
2. Pose estimation using mediapipe.
3. Pose estimation using DWPose.
4. Add biblical metadata like verse reference range, list of BibleNLP vrefs and parallel Bible verse text(from BSB version).
5. Also add more video specific info to metadata like number of frames, fps, width, height etc.

**Outputs**:
For each sample:
1. mp4 video
1. .pose file
1. .npz file
1. transcript.json
1. .json


![dataflow](./docs/ISLDataprepArch.jpg)


