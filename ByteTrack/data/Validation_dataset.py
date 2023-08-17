# IRP ByteTracker

import os

import numpy as np
import pandas as pd
import cv2

import torch
import torch.utils.data as data


class ValidationDataset(data.Dataset):
    """ByteTrack Validation Dataset."""

    def __init__(self, data_path: str, label_path: str):
        """Initialise the dataset and raise FileNotFoundError if folders not found."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")

        self.data_path = data_path
        self.label_path = label_path

        # Get the information about the input source
        self.data_info, self.fps = self.get_data_info()

    def __getitem__(self, index):
        """Returns next frame path, frame, and targets."""
        frame_path = self.data_info["frame_path"][index]
        label_path = self.data_info["label_path"][index]

        # Load the frame
        frame = cv2.imread(frame_path) # BGR
        height, width, _ = frame.shape

        bboxes = []
        labels = []
        ids = []

        # Read the different object annotated for the image
        with open(label_path, 'r') as f:
            lines = [line[:-1] if line[-1] == "\n" else line for line in f.readlines()]
        
        # Loop through the annotations and resize the boxes
        # The boxes need to be in YOLO format: normalised xywh
        
        for line in lines:
            line = line.split(' ')
            ids.append(int(line[0])) # Tracking IDs
            labels.append(int(line[1])) # Objects labels
            box = [float(val) for val in line[2:]] # Bounding boxes in YOLO format (normalised xywh)

            box[0] = int(box[0] * width)
            box[2] = int(box[2] * width)

            box[1] = int(box[1] * height)
            box[3] = int(box[3] * height)
            
            box = [int(np.floor(np.max((0, box[0] - box[2]/2)))), int(np.floor(np.max((0, box[1] - box[3]/2)))), int(np.ceil(np.min((width, box[0] + box[2]/2)))), int(np.ceil(np.min((height, box[1] + box[3]/2))))]
            
            # Update box location if one of the box dimension is equal to 0
            if box[0] == box[2]:
                if box[0] != 0:
                    box[0] -= 1
                
                if box[2] != width:
                    box[2] += 1
            
            if box[1] == box[3]:
                if box[1] != 0:
                    box[1] -= 1
                
                if box[3] != height:
                    box[3] += 1
            
            bboxes.append(box)

        target = {"boxes": torch.Tensor(bboxes), "labels": torch.tensor(labels, dtype=torch.int64), 'ids': torch.tensor(ids, dtype=torch.int64)}
        
        return frame_path, frame, target

    def __len__(self):
        """Returns the number of frames in the source."""
        return len(self.data_info)
    
    def get_data_info(self):
        """
        Get the info about each video sequences.
        
        Args: 
            None

        Returns: 
            pd.DataFrame: It contains the direction of each video sequences, the direction of each annotation folder 
                          and the fps of the video sequence
        """

        # Get the path of each frame and annotation file in the frame sequence
        files = [os.path.join(self.data_path,file.replace(".txt", ".jpg")) for file in sorted(os.listdir(self.label_path))]
        labels = [os.path.join(self.label_path,file) for file in sorted(os.listdir(self.label_path))]

        # Read the meta_info.txt file of each video sequences to get the fps value
        with open(os.path.join(self.data_path, "meta_info.txt"), 'r') as f:
                lines = f.readlines()
                fps = int(lines[-3].split(' ')[-1][:-1] if lines[-3].split(' ')[-1][-1] == "\n" else lines[-3].split(' ')[-1])
                    
        
        return pd.DataFrame({
            'frame_path': files,
            'label_path': labels
        }), fps