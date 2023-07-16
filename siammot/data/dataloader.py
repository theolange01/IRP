import os
from typing import Dict, Union

import numpy as np

from torch.utils.data import DataLoader

from .video_dataset import VideoDataset, VideoDatasetBatchCollator
from .image_dataset import ImageDataset, ImageDatasetBatchCollator
from .augmentation.build_augmentation import build_siam_augmentation

import albumentations as A

from siammot.utils import LOGGER, colorstr

def build_dataset(cfg, source: str) -> Dict[str, Union[VideoDataset, ImageDataset]]:
    """
    Build the Dataset needed during training.

    Args: 
        cfg object
        (str) source: Path to the folder where the data are stored in the format (Train, ) or (Train, Val)

    Returns: 
        Dataset dict
    """

    dataset_list = source # (Train, ) or (Train, Val)
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    datasets = []
    for index, dataset_key in enumerate(dataset_list):
        
        if cfg.MODEL.TRACK_ON:
            transforms = build_siam_augmentation(cfg, is_train = (index==0))
            
            # create the VideoDataset
            _dataset = VideoDataset(os.path.join(dataset_key, "data"),
                                    os.path.join(dataset_key, 'labels'),
                                    sampling_interval=cfg.VIDEO.TEMPORAL_SAMPLING,
                                    clip_len=cfg.VIDEO.TEMPORAL_WINDOW,
                                    transforms=transforms,
                                    frames_in_clip=cfg.VIDEO.RANDOM_FRAMES_PER_CLIP)
        else:
        
            transforms = A.Compose([A.ColorJitter(),
                                    A.Blur(),
                                    A.HorizontalFlip(p=0.5)], 
                                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.2))
        
            _dataset = ImageDataset(os.path.join(dataset_key, "data"),
                                    os.path.join(dataset_key, 'labels'),
                                    transforms=transforms)
        

        datasets.append(_dataset)


        if index == 0:
            LOGGER.info(f"{colorstr('train:')} {len(_dataset)} clips created ✅")
        else:
            LOGGER.info(f"{colorstr('val:')} {len(_dataset)} clips created ✅")

    if len(dataset_list) == 2:
        dataset = {'train': datasets[0],
                'val': datasets[1]}
    else:
        dataset = {'train': datasets[0],
                'val': None}

    return dataset


def build_train_data_loader(cfg, source: str, batch_size: int, num_workers: int =2) -> Dict[str, DataLoader]:
    """
    Build the DataLoaders needed for training given the configuration file and the source of the data

    Args:
        cfg object
        source (str): Path to the folder where the data are stored in the format (Train, ) or (Train, Val)
        batch_size (int) : The number of data in each batch
        num_workers (int)

    Returns: 
        DataLoader Dict
    """

    dataset = build_dataset(cfg, source)

    # Build train and val dataset and put it in dict
    if cfg.MODEL.TRACK_ON:
      if dataset['val']:
          data_loaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=VideoDatasetBatchCollator()),
                          'val': DataLoader(dataset['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=VideoDatasetBatchCollator())}
      
      else:
          data_loaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=VideoDatasetBatchCollator()),
                          'val': None}
    
    else:
      if dataset['val']:
          data_loaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ImageDatasetBatchCollator()),
                          'val': DataLoader(dataset['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ImageDatasetBatchCollator())}
      
      else:
          data_loaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ImageDatasetBatchCollator()),
                          'val': None}

    return data_loaders