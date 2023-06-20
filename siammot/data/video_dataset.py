import os
import random
from glob import glob

import pandas as pd

import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor

from PIL import Image


class VideoDataset(data.Dataset):
    """
    VideoDataset.

    Args:
        
    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.
    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
    Returns:
        VideoDataset: The Dataset used to load videos for training.
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

        self.clips_list = glob(os.listdir(self.data_path))
        
        self.clip_len = clip_len
        self.transforms = transforms
        self.frames_in_clip = min(clip_len, frames_in_clip)

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
    
    def get_frame(self, video_path, frame_idx):
        lst_frames = sorted(glob(os.path.join(video_path, "*.jpg")))

        image = Image.open(lst_frames[frame_idx]).convert('RGB')

        return to_tensor(image)
    
    def get_anno(self, anno_path, frame_idx, img_shape):
        _, width, height = img_shape
        lst_anno = sorted(os.listdir(anno_path))
        boxes = []
        labels = []

        with open(lst_anno[frame_idx], 'r') as f:
            lines = [line[:-1] for line in f.readlines()]
        
        for line in lines:
            line = line.split(' ')
            labels.append(line[0])
            box = [float(val) for val in line[1:]] # Need to convert Annotations to xyxy, originally in xywh normalised

            box[0] = box[0] * width
            box[2] = box[2] * width

            box[1] = box[1] * height
            box[3] = box[3] * height

            boxes.append([box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2])

        return {'boxes': torch.Tensor(boxes), 'labels': torch.Tensor(labels)} # 'ids': torch.Tensor([-1]*len(labels)) see if needed


    
    def get_video_clips(self, sampling_interval_ms):
        """
        Process the long videos to a small video chunk (with self.clip_len seconds)
        Video clips are generated in a temporal sliding window fashion
        """
        
        video_clips = []
        for index in range(len(self.data_info)):
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
        files = [os.path.join(self.data_path,file) for file in sorted(os.listdir(self.data_path))]
        labels = [os.path.join(self.label_path,file) for file in sorted(os.listdir(self.label_path))]
        fps = []

        for file in files:
            with open(os.path.join(file, "meta_info.ini"), 'r') as f:
                    fps.append(int(f.readlines()[-3].split(' ')[-1][:-1])) # todo: precise position of the info
                    
        
        return pd.DataFrame({
            'file_path': files,
            'label_path': labels,
            'fps': fps
        })

    


"""
    def get_video_clips_V0(self, sampling_interval_ms):
        
        Process the long videos to a small video chunk (with self.clip_len seconds)
        Video clips are generated in a temporal sliding window fashion
        
        
        video_clips = []
        for index, video in enumerate(self.videos):
            frame_idxs_with_anno = []
            for i in range(len(video[0])):
                if len(video[1][i]['labels']) == 0:
                    frame_idxs_with_anno.append(i)
            
            if len(frame_idxs_with_anno) == 0:
                continue
            # The video clip may not be temporally continuous
            start_frame = min(frame_idxs_with_anno)
            end_frame = max(frame_idxs_with_anno)

            # make sure that the video clip has at least two frames
            clip_len_in_frames = max(self.frames_in_clip, int(self.clip_len / 1000. * video[2]))
            sampling_interval = int(sampling_interval_ms / 1000. * video[2])

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

def load_videos(self):
        videos = []

        clips_list = glob(os.listdir(self.data_path))
        for clip_path in clips_list:           
            images = []
            if len(glob(os.path.join(clip_path, "*.mp4"))):
                capture = cv2.VideoCapture(glob(os.path.join(clip_path, "*.mp4"))[0])
                fps = capture.get(cv2.CAP_PROP_FPS)

                while True:
                    ret, frame = capture.read()

                    if frame is None:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)

                    images.append(frame)

            elif len(glob(os.path.join(clip_path, "*.avi"))):
                capture = cv2.VideoCapture(glob(os.path.join(clip_path, "*.avi"))[0])
                fps = capture.get(cv2.CAP_PROP_FPS)

                while True:
                    ret, frame = capture.read()
                    if frame is None:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    
                    images.append(frame)
            else:
                lst_images = sorted(glob(os.path.join(clip_path, '*.jpg')))
                with open(os.path.join(clip_path, "meta_info.ini"), 'r') as f:
                    fps = int(f.readlines()[-4].split(' ')[-1][:-1]) # todo: precise position of the info

                for img_path in lst_images:
                    frame = Image.open(img_path).convert('RGB')

                    images.append(frame)

            targets = [[{'boxes': torch.Tensor([]), 'labels': torch.Tensor([])}] for _ in range (len(images))]
            with open(os.path.join(clip_path, 'labels.txt'), 'r') as f:
                lines = [line[:-1] for line in f.readlines()]
            
            for line in lines:
                frame_id = int(line[0])
                label = torch.Tensor([int(line[1])])
                bbox = torch.Tensor([[float(line[i]) for i in range(2,6)]])
                targets[frame_id]['boxes'] = torch.cat((targets[frame_id]['boxes'], bbox)) # xyxy format
                targets[frame_id]['labels'] = torch.cat((targets[frame_id]['labels'], label))

            videos.append((images, targets, fps))

        return videos

"""