import sys
import vpi
import numpy as np
from os import path
from argparse import ArgumentParser
from contextlib import contextmanager
import cv2
import time

# --------------------------------------
# Some definitions and utility functions

# Maximum number of keypoints that will be tracked
MAX_KEYPOINTS = 100

def update_mask(mask, trackColors, prevFeatures, curFeatures, status = None):
    '''Draw keypoint path from previous frame to current one'''

    numTrackedKeypoints = 0

    def none_context(a=None): return contextmanager(lambda: (x for x in [a]))()

    with curFeatures.rlock_cpu(), \
         (status.rlock_cpu() if status else none_context()), \
         (prevFeatures.rlock_cpu() if prevFeatures else none_context()):

        for i in range(curFeatures.size):
            # keypoint is being tracked?
            if not status or status.cpu()[i] == 0:
                color = tuple(trackColors[i,0].tolist())

                # OpenCV 4.5+ wants integers in the tuple arguments below
                cf = tuple(np.round(curFeatures.cpu()[i]).astype(int))

                # draw the tracks
                # if prevFeatures:
                #     pf = tuple(np.round(prevFeatures.cpu()[i]).astype(int))
                #     cv2.line(mask, pf, cf, color, 2)

                cv2.circle(mask, cf, 5, color, -1)

                numTrackedKeypoints += 1

    return numTrackedKeypoints

def save_file_to_disk(frame, mask, baseFileName, frameCounter):
    '''Apply mask on frame and save it to disk'''

    frame = frame.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)
    with frame.rlock_cpu() as frameData:
        frame = cv2.add(frameData, mask)

    # name, ext = path.splitext(baseFileName)
    # fname = "{}_{:04d}{}".format(name, frameCounter, ext)

    cv2.imshow("xxx", frame)
    cv2.waitKey(100)

# ----------------------------
# main
# ----------------------------

backend = vpi.Backend.CUDA
# ----------------
# Open input video

inVideo = cv2.VideoCapture("/home/user/projects/of_vio/scripts/dashcam.mp4")

# Read first input frame
ok, cvFrame = inVideo.read()
if not ok:
    exit('Cannot read first input frame')

# ---------------------------
# Perform some pre-processing

# Retrieve features to be tracked from first frame using
# Harris Corners Detector
with vpi.Backend.CPU:
    frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8)
    curFeatures, scores = frame.harriscorners(strength=0.1, sensitivity=0.01)

# Limit the number of features we'll track and calculate their colors on the
# output image
with curFeatures.lock_cpu() as featData, scores.rlock_cpu() as scoresData:
    print(curFeatures.size, scoresData.shape)
    sys.exit()
    # Sort features in descending scores order and keep the first MAX_KEYPOINTS
    ind = np.argsort(scoresData, kind='mergesort')[::-1]
    featData[:] = np.take(featData, ind, axis=0)
    curFeatures.size = min(curFeatures.size, MAX_KEYPOINTS)

    # Keypoints' have different hues, calculated from their position in the first frame
    trackColors = np.array([[(int(p[0]) ^ int(p[1])) % 180,255,255] for p in featData], np.uint8).reshape(-1,1,3)
    # Convert colors from HSV to RGB
    trackColors = cv2.cvtColor(trackColors, cv2.COLOR_HSV2BGR).astype(int)

with backend:
    optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, 4)

# Counter for the frames
idFrame = 0


while True:
    prevFeatures = curFeatures

    # Read one input frame
    ret, cvFrame = inVideo.read()
    if not ret:
        print("Video ended.")
        break
    idFrame += 1

    # Convert frame to grayscale
    with vpi.Backend.CUDA:
        frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8);

        # Calculate where keypoints are in current frame
        curFeatures, status = optflow(frame)

    # Update the mask with the current keypoints' position
    # mask = np.zeros((frame.height, frame.width, 3), np.uint8)
    # numTrackedKeypoints = update_mask(mask, trackColors, prevFeatures, curFeatures, status)

    # # No more keypoints to track?
    # if numTrackedKeypoints == 0:
    #     print("No keypoints to track.")
    #     break # nothing else to do

    # Apply mask to frame and save it to disk
    # save_file_to_disk(frame, mask, "", idFrame)
    time.sleep(1/10)