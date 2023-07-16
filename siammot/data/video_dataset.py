import os
import random
from glob import glob
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict

import pandas as pd

import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor

from siammot.data import to_image_list

from PIL import Image


class VideoDataset(data.Dataset):
    """
    VideoDataset. Create a Dataset for the training of the SiamMOT model.
    It creates different video clips.
    """

    def __init__(self, data_path: str, label_path: str, sampling_interval=250, clip_len=1000,
                 is_train=True, frames_in_clip=2, transforms=None):
        """
        Initializes the YOLO model.
        Args:
            data_path (str): Path to the data folder. This folder must contains videos or images sequences.
            label_path (str): Path to the label folder. It must contains the same subfolders as the data_path folder and a .txt file 
                              for each images from the image sequences with the same name
            sampling_interval (int): the temporal stride (in ms) of sliding window
            clip_len (int): the temporal length (in ms) of video clips
            is_train (Bool): a boolean flag indicating whether it is training
            frames_in_clip (int): the number of frames sampled in a video clip (for a training example)
            transforms (Any): frame-level transformation before they are fed into neural networks
        """

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
        
        assert is_train is True, "The dataset class only supports training and validation"
        assert (2 >= frames_in_clip > 0), "frames_in_clip has to be 1 or 2"

        self.data_path = data_path
        self.label_path = label_path

        self.clips_list = glob(os.path.join(self.data_path, "*/"))
        
        self.clip_len = clip_len
        self.transforms = transforms
        self.frames_in_clip = min(clip_len, frames_in_clip)

        # Get the different information needed to create the video clips
        self.data_info = self.get_data_info()

        # Process dataset to get all valid video clips
        self.clips = self.get_video_clips(sampling_interval_ms=sampling_interval)


    def __getitem__(self, index):
        video = []
        target = []

        (sample_id, clip_frame_ids) = self.clips[index]
        video_path = self.data_info['file_path'][sample_id]
        anno_path = self.data_info['label_path'][sample_id]

        # Randomly sampling self.frames_in_clip frames
        # And keep their relative temporal order
        rand_idxs = sorted(random.sample(clip_frame_ids, self.frames_in_clip))
        for frame_idx in rand_idxs:
            im = self.get_frame(video_path, frame_idx)
            anno = self.get_anno(anno_path, frame_idx, im.shape)

            video.append(im)
            target.append(anno)

        # Video clip-level augmentation
        if self.transforms is not None:
            video, target = self.transforms(video, target)

        return video, target, sample_id


    def __len__(self):
        return len(self.clips)
    
    def get_frame(self, video_path: str, frame_idx: int) -> torch.Tensor:
        """
        
        Args: 
            video_path (str): The path to the folder containing the requested frame
            frame_idx (int): The index of the frame to load

        Returns: 
            frame (Tensor): The loaded frame
        """
        
        # Get the list of frames
        lst_frames = sorted(glob(os.path.join(video_path, "*.jpg")))

        # Load the frame
        image = Image.open(lst_frames[frame_idx]).convert('RGB')

        return to_tensor(image)
    
    def get_anno(self, anno_path: str, frame_idx: int, img_shape: Tuple) -> Dict[str, torch.Tensor]:
        # Get the shape of the image to resize the boxes
        _, height, width = img_shape

        # Get the list of annotation files
        lst_anno = sorted(os.listdir(anno_path))
        boxes = []
        labels = []

        # Read the different object annotated for the image
        with open(os.path.join(anno_path, lst_anno[frame_idx]), 'r') as f:
            lines = [line[:-1] if line[-1] == "\n" else line for line in f.readlines()]
        
        # Loop through the annotations and resize the boxes
        # The boxes need to be in YOLO format: normalised xywh
        for line in lines:
            line = line.split(' ')
            labels.append(int(line[0]))
            box = [float(val) for val in line[1:]] # Need to convert Annotations to xyxy, originally in xywh normalised

            box[0] = box[0] * width
            box[2] = box[2] * width

            box[1] = box[1] * height
            box[3] = box[3] * height

            boxes.append([int(box[0] - box[2]/2), int(box[1] - box[3]/2), int(box[0] + box[2]/2), int(box[1] + box[3]/2)])

        return {'boxes': torch.Tensor(boxes), 'labels': torch.tensor(labels, dtype=torch.int64)} # 'ids': torch.Tensor([-1]*len(labels)) see if needed


    
    def get_video_clips(self, sampling_interval_ms: int):
        """
        Process the long videos to a small video chunk (with self.clip_len seconds)
        Video clips are generated in a temporal sliding window fashion
        """
        
        video_clips = []
        # Loop through all the different video sequences
        for index in range(len(self.data_info)):

            # Get the index of the frames that contains annotations
            frame_idxs_with_anno = []
            lst_anno = sorted(glob(os.path.join(self.data_info["label_path"][index], "*.txt")))
            for i in range(len(lst_anno)):
                with open(lst_anno[i], 'r') as f:
                    if len(f.readlines()) != 0:
                        frame_idxs_with_anno.append(i)
            
            if len(frame_idxs_with_anno) == 0:
                continue
            # The video clip may not be temporally continuous
            start_frame = min(frame_idxs_with_anno)
            end_frame = max(frame_idxs_with_anno)

            # make sure that the video clip has at least two frames
            clip_len_in_frames = max(self.frames_in_clip, int(self.clip_len / 1000. * self.data_info['fps'][index]))
            sampling_interval = int(sampling_interval_ms / 1000. * self.data_info['fps'][index])

            # Sample video clips
            for idx in range(start_frame, end_frame, sampling_interval):
                clip_frame_ids = []

                # only include frames with annotation within the video clip
                for frame_idx in range(idx, idx + clip_len_in_frames):
                    if frame_idx in frame_idxs_with_anno:
                        clip_frame_ids.append(frame_idx)

                # Only include video clips that have at least self.frames_in_clip annotating frames
                if len(clip_frame_ids) >= self.frames_in_clip:
                    video_clips.append((index, clip_frame_ids))

        return video_clips
    
    def get_data_info(self):
        """
        Get the info about each video sequences
        
        Args: 
            None

        Returns: 
            pd.DataFrame: It contains the direction of each video sequences, the direction of each annotation folder 
                          and the fps of the video sequence
        """

        files = [os.path.join(self.data_path,file) for file in sorted(os.listdir(self.data_path))]
        labels = [os.path.join(self.label_path,file) for file in sorted(os.listdir(self.label_path))]
        fps = []

        # Read the meta_info.txt file of each video sequences to get the fps value
        for file in files:
            with open(os.path.join(file, "meta_info.txt"), 'r') as f:
                    lines = f.readlines()
                    fps.append(int(lines[-3].split(' ')[-1][:-1] if lines[-3].split(' ')[-1][-1] == "\n" else lines[-3].split(' ')[-1])) # todo: precise position of the info
                    
        
        return pd.DataFrame({
            'file_path': files,
            'label_path': labels,
            'fps': fps
        })


class VideoDatasetBatchCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        image_batch = list(itertools.chain(*transposed_batch[0]))
        # image_batch = to_image_list(image_batch)

        # to make sure that the id of each instance
        # are unique across the whole batch
        targets = transposed_batch[1]
        video_ids = transposed_batch[2]
        uid = 0
        video_id_map = defaultdict(dict)
        for targets_per_video, video_id in zip(targets, video_ids):
            for targets_per_video_frame in targets_per_video:
                if 'ids' in targets_per_video_frame.keys():
                    _ids = targets_per_video_frame['ids']
                    _uids = _ids.clone()
                    for i in range(len(_ids)):
                        _id = _ids[i].item()
                        if _id not in video_id_map[video_id]:
                            video_id_map[video_id][_id] = uid
                            uid += 1
                        _uids[i] = video_id_map[video_id][_id]
                    targets_per_video_frame['ids'] = _uids

        targets = list(itertools.chain(*targets))

        return image_batch, targets, video_ids
