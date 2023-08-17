# IRP ByteTracker

import os
import cv2
import torch.utils.data as data

from .utils import VID_FORMATS


class TrackingDataset(data.Dataset):
    """ByteTrack Tracking Dataset."""

    def __init__(self, data_path: str):
        """Initialise the dataset and raise FileNotFoundError if file not found."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
        
        self.data_path = data_path
        self.is_video = data_path.endswith(VID_FORMATS) # The source can be a single frame, a frame sequence or a video
        self.capture = None

        # Get the information about the input source
        self.data_info, self.fps = self.get_data_info()

    def __getitem__(self, index):
        """Returns next frame path and frame."""
        frame_path = self.data_info[index]

        if self.is_video: # The source is a video file
            # Set the current frame to the needed one
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)

            # Read the frame
            ret, frame = self.capture.read()
            assert ret

            # Release the video is the last frame has been read
            if index == len(self.data_info)-1:
                self.capture.release()

        else: # The source corresponds to a single frame or a sequence of frame
            frame = cv2.imread(frame_path) # BGR
        
        return frame_path, frame

    def __len__(self):
        """Returns the number of frames in the source."""
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

            # Read the meta_info.txt file of each video sequences to get the fps value
            with open(os.path.join(self.data_path, "meta_info.txt"), 'r') as f:
                    lines = f.readlines()
                    fps = (int(lines[-3].split(' ')[-1][:-1] if lines[-3].split(' ')[-1][-1] == "\n" else lines[-3].split(' ')[-1])) # todo: precise position of the info
        
        else:
            self.capture = cv2.VideoCapture(self.data_path)
            fps = self.capture.get(cv2.CV_CAP_PROP_FPS) # Get the video FPS
            files = ['image0.jpg' for _ in self.capture.get(cv2.CAP_PROP_FRAME_COUNT)] # Create a default path for each frame of the video
        
        return files, fps