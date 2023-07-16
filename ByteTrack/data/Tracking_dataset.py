import os

import numpy as np
import pandas as pd
import cv2

import torch
import torch.utils.data as data

from .utils import VID_FORMATS


class TrackingDataset(data.Dataset):
    """
    
    
    """

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
        
        self.data_path = data_path
        self.is_video = data_path.endswith(VID_FORMATS)
        self.capture = None

        self.data_info = self.get_data_info()
        self.fps = self.data_info['fps'][0]

    def __getitem__(self, index):
        frame_path = self.data_info["frame_path"][index]

        if self.is_video:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)

            ret, frame = self.capture.read()
            assert ret

            if index == len(self.data_info)-1:
                self.capture.release()
        else:
            frame = cv2.imread(frame_path)
        
        return frame_path, frame

    def __len__(self):
        return len(self.data_info)
    
    def get_data_info(self):
        """
        Get the info about each video sequences
        
        Args: 
            None

        Returns: 
            pd.DataFrame: It contains the direction of each video sequences, the direction of each annotation folder 
                          and the fps of the video sequence
        """

        if not self.is_video:
            files = [os.path.join(self.data_path,file) for file in sorted(os.listdir(self.data_path))[:-1]]
            fps = []

            # Read the meta_info.txt file of each video sequences to get the fps value
            with open(os.path.join(self.data_path, "meta_info.txt"), 'r') as f:
                    lines = f.readlines()
                    fps.append(int(lines[-3].split(' ')[-1][:-1] if lines[-3].split(' ')[-1][-1] == "\n" else lines[-3].split(' ')[-1])) # todo: precise position of the info
        
        else:
            self.capture = cv2.VideoCapture(self.data_path)
            fps = self.capture.get(cv2.CV_CAP_PROP_FPS)
            files = ['image0.jpg' for _ in self.capture.get(cv2.CAP_PROP_FRAME_COUNT)]
        
        return pd.DataFrame({
            'frame_path': files,
            'fps': fps
        })