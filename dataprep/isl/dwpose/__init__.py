# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from numpy.linalg import LinAlgError
import math
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    # faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    # canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg, normalize=False):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            if normalize:
                candidate = np.array([transform_coordinates(candidate[0])])
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1


            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            # pose = dict(bodies=bodies, hands=hands, faces=faces)
            pose = dict(bodies=bodies, hands=hands)

            return draw_pose(pose, H, W)

def solve_affine( p1, p2, p3, s1, s2, s3):
    x = np.transpose(np.matrix([p1,p2,p3]))
    y = np.transpose(np.matrix([s1,s2,s3]))
    # add ones on the bottom of x and y
    x = np.vstack((x,[1,1,1]))
    y = np.vstack((y,[1,1,1]))
    # solve for A2
    try:
        A2 = y * x.I
    except LinAlgError:
        # print(x)
        epsilon=0.0001
        x[2,0] += epsilon
        x[2,3] += epsilon
        # print(x)
        A2 = y * x.I
       
    # return function that takes input x and transforms it
    # don't need to return the 4th row as it is
    return lambda x: (A2*np.vstack((np.matrix(x).reshape(2,1),1)))[0:2,:]


def transform_coordinates(results):
    shoulder_left = results[5]
    hip_left = results[11]
    hip_right = results[8]
    new_hip_left=[0.67965056, 0.8621226 ]
    trans = [hip_left[0] -new_hip_left[0], hip_left[1]-new_hip_left[1]]
    orig_hip_len = math.sqrt((hip_left[0] - hip_right[0]) ** 2 + \
                               (hip_left[1] - hip_right[1]) ** 2
                           )
    orig_shoulder_len = math.sqrt((hip_left[0] - shoulder_left[0]) ** 2 + \
                               (hip_left[1] - shoulder_left[1]) ** 2
                           )
    new_hip_len = 0.18984542333244844
    new_shoulder_len = 0.39029887344139114

    scale_hip = new_hip_len/orig_hip_len
    scale_shoulder = new_shoulder_len/orig_shoulder_len
    new_shoulder_left = [shoulder_left[0]-trans[0], shoulder_left[1]-trans[1]] #after translation
    new_shoulder_left = [new_hip_left[0] + (new_shoulder_left[0]-new_hip_left[0])*scale_shoulder,
                          new_hip_left[1] + (new_shoulder_left[1]-new_hip_left[1])*scale_shoulder]
    new_hip_right = [hip_right[0]-trans[0], hip_right[1]-trans[1]] #after translation
    new_hip_right = [new_hip_left[0] + (new_hip_right[0]-new_hip_left[0])*scale_hip,
                          new_hip_left[1] + (new_hip_right[1]-new_hip_left[1])*scale_hip]
    primary_system1 = shoulder_left #left Shoulder
    primary_system3 = hip_left #left Hip
    primary_system4 = hip_right #right Hip
    secondary_system1 = new_shoulder_left #left Shoulder
    secondary_system3 = new_hip_left #left Hip
    secondary_system4 = new_hip_right #right Hip
    transformFn = solve_affine( primary_system1,
        primary_system3, primary_system4,
        secondary_system1,
        secondary_system3, secondary_system4 )
    new_result = []
    for landmark in results:
        if landmark[0] == -1:
            new_result.append(landmark)
        else:
            new = transformFn(landmark)
            new_result.append(new)

    return new_result
