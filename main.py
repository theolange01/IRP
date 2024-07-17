import os
import sys
from glob import glob

import cv2
from ByteTrack import Tracker
from ByteTrack.utils.Annotator import Annotator

model = Tracker(model="ByteTrack/weights/best.pt")

# Get the path to a video or image sequence
data_path = "IRP_dataset/test/Test_13"

# Get the fps saved in the same folder
with open(os.path.join(data_path, "meta_info.txt"), "r") as f:
    meta_info = [line[:-1] if line[-1] == '\n' else line for line in f.readlines()]
    
fps = int(meta_info[4].split(" ")[-1])

# List of frames
lst_frames = sorted(glob(os.path.join(data_path, '*.jpg')))
ObjectAnnotator = Annotator()

for file in lst_frames:
    frame = cv2.imread(file)

    results = model.track(frame, persist=True, save_video=False, save_txt=False, save_frame=False, verbose=False, fps=fps)

    frame = ObjectAnnotator(frame, results[0])

    cv2.imshow("Tracking Results", frame)
    key = cv2.waitkey(60)
    if key == ord("s"):
        cv2.waitkey(0)