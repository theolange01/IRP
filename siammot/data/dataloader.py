import os

import numpy as np

from torch.utils.data import DataLoader

from .video_dataset import VideoDataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from siammot.utils import LOGGER

def build_dataset(cfg, source):
    """

    """

    dataset_list = source # (Train, ) or (Train, Val)
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    datasets = []
    for index, dataset_key in enumerate(dataset_list):

        transforms = GeneralizedRCNNTransform(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                              image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])


        _dataset = VideoDataset(os.path.join(dataset_key, "data"),
                                os.path.join(dataset, 'labels'),
                                sampling_interval=cfg.VIDEO.TEMPORAL_SAMPLING,
                                clip_len=cfg.VIDEO.TEMPORAL_WINDOW,
                                transforms=transforms,
                                frames_in_clip=cfg.VIDEO.RANDOM_FRAMES_PER_CLIP)
        
        datasets.append(_dataset)
        if index == 0:
            LOGGER.info(f"Training dataset created: {len(_dataset)} clips created")
        else:
            LOGGER.info(f"Validation dataset created: {len(_dataset)} clips created")

    if len(dataset_list) == 2:
        dataset = {'train': datasets[0],
                'val': datasets[1]}
    else:
        dataset = {'train': datasets[0],
                'val': None}

    return dataset


def build_train_data_loader(cfg, source, batch_size, num_workers=8):

    dataset = build_dataset(cfg, source)
    # Build train and val dataset and put it in dict
    if dataset['val']:
        data_loaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
                        'val': DataLoader(dataset['val'], batch_size=batch_size, shuffle=True, num_workers=num_workers)}
    
    else:
        data_loaders = {'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)}

    return data_loaders