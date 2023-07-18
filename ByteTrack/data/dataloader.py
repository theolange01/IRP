import os
from glob import glob
from pathlib import Path

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from .utils import IMG_FORMATS, VID_FORMATS

from PIL import Image

from .tracking_dataset import TrackingDataset
from .validation_dataset import ValidationDataset


class Collator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):        
        return batch

def load_track_dataloader(source):
    if isinstance(source, np.ndarray):
        try:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        except:
            pass
        return {'data': [(None, source)], 'fps': None, "type": "frame"}
    elif isinstance(source, Image):
        try:
            source = cv2.cvtColor(np.array(source), cv2.COLOR_BGR2RGB)
        except:
            source = np.array(source)
        return {'data': [(None, source)], 'fps': None, "type": "frame"}
    elif isinstance(source, torch.Tensor):
        source = source.numpy().transpose(1, 2, 0)
        return {'data': [(None, source)], 'fps': None, "type": "frame"}
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"The given path '{source}' does not exists or is wrong")

        if source.split('.')[-1] in IMG_FORMATS:
            return {'data': [source, cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)], 'fps': None, "type": "frame"}
        
        else:

            _dataset = TrackingDataset(source)
            fps = _dataset.fps
            _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=Collator())
            DataLoaders = {"data": _dataloader, 'fps': fps, "type": "video"}


            return DataLoaders
    else:
        raise TypeError('Unsupported Source type.')


def load_validation_dataloader(source):
    if not os.path.exists(source):
        raise FileNotFoundError(f"The given path '{source}' does not exists or is wrong")
    
    lst_videos = os.listdir(os.path.join(source, "data"))

    DataLoaders = []

    for video in lst_videos:
        _dataset = ValidationDataset(os.path.join(source, "data", video), os.path.join(os.path.join(source, "labels", video)))
        fps = _dataset.fps
        _dataloader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=Collator())
        DataLoaders.append({"data": _dataloader, 'fps': fps})


    return DataLoaders