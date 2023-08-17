# IRP ByteTracker

import os

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader

from .utils import IMG_FORMATS
from .tracking_dataset import TrackingDataset
from .validation_dataset import ValidationDataset


class ValidationCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the Validation DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        return transposed_batch[0][0], transposed_batch[1][0], transposed_batch[2][0]


class TrackingCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the Tracking DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        return transposed_batch[0][0], transposed_batch[1][0]
    

def load_track_dataloader(source):
    """
    Create the tracking dataloader
    
    Args:
        source (str, np.ndarray, PIL.Image, tuple, list): Frames to use for tracking. 
                                                          It can be a single frame, a frame sequence or a video.

    Returns:
        Dict containing:
            data: Tracking Dataloader
            fps (int): video speed
            type (str): Type of data. Can be 'frame' or 'video'   

    Note:
        The colour format of the input frames is important. Once converted to OpenCV images, it should be in BGR format  
    """

    # Verify the input source and return the corresponding tracking dataloader
    if isinstance(source, np.ndarray):
        # Single frame given as input as an numpy array (OpenCV Frame)
        # The image has to be in the BGR colour format
        return {'data': [(None, source)], 'fps': None, "type": "frame"}
    
    elif isinstance(source, Image.Image):
        # Single frame given as input as a PIL Image. The PIL Image should be in RGB format
        source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR) # BGR
        return {'data': [(None, source)], 'fps': None, "type": "frame"}
    
    elif isinstance(source, torch.Tensor):
        # Single frame given as input as a torch Tensor
        source = source.numpy().transpose(1, 2, 0) # RGB
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR) # BGR
        return {'data': [(None, source)], 'fps': None, "type": "frame"}
    
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"The given path '{source}' does not exists or is wrong")

        if source.split('.')[-1] in IMG_FORMATS:
            # Single frame given by its directory
            return {'data': [source, cv2.imread(source)], 'fps': None, "type": "frame"} # Frame in BGR
        
        else:
            # An entire video or image sequences
            _dataset = TrackingDataset(source)
            fps = _dataset.fps
            _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=TrackingCollator())
            DataLoaders = {"data": _dataloader, 'fps': fps, "type": "video"}

            return DataLoaders
    else:
        raise TypeError('Unsupported Source type.')



def load_validation_dataloader(source):
    """
    Create the validating dataloader
    
    Args:
        - source (str, np.ndarray, PIL.Image, tuple, list): Frames to use for tracking. 
                                                            It can be a single frame, a frame sequence or a video.

    Returns:
        - Dict containing:
            - data: Validation Dataloader
            - fps (int): video speed
            - type (str): Type of data. Can be 'frame' or 'video'     
    """


    if not os.path.exists(source):
        raise FileNotFoundError(f"The given path '{source}' does not exists or is wrong")
    
    # Get the list of videos in the validating folder
    lst_videos = os.listdir(os.path.join(source, "data"))

    DataLoaders = []

    for video in lst_videos:
        # Create a DataLoader for each validating video
        _dataset = ValidationDataset(os.path.join(source, "data", video), os.path.join(os.path.join(source, "labels", video)))
        fps = _dataset.fps
        _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=ValidationCollator())
        DataLoaders.append({"data": _dataloader, 'fps': fps})


    return DataLoaders