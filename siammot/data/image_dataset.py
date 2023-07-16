import os
import time
from glob import glob
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor

class ImageDataset(data.Dataset):
    def __init__(self, data_path: str, label_path: str, transforms=None):
    
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
            
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"The given path '{data_path}' does not exists or is wrong")
    
        self.data_path = data_path
        self.label_path = label_path
    
        self.clips_list = glob(os.path.join(self.data_path, "*/"))
        
        self.transforms = transforms
    
        # Get the different information needed to load the images and targets
        self.data_info = self.get_data_info()
    
    def __getitem__(self, index):
        image_path = self.data_info["image_path"][index]
        anno_path = self.data_info["labels_path"][index]
        
        # Load the frame
        image =  cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        bboxes = []
        labels = []

        # Read the different object annotated for the image
        with open(anno_path, 'r') as f:
            lines = [line[:-1] if line[-1] == "\n" else line for line in f.readlines()]
        
        # Loop through the annotations and resize the boxes
        # The boxes need to be in YOLO format: normalised xywh
        
        for line in lines:
            line = line.split(' ')
            labels.append(1+int(line[0]))
            box = [float(val) for val in line[1:]] # Need to convert Annotations to xyxy, originally in xywh normalised

            box[0] = int(box[0] * width)
            box[2] = int(box[2] * width)

            box[1] = int(box[1] * height)
            box[3] = int(box[3] * height)
            
            box = [int(np.floor(np.max((0, box[0] - box[2]/2)))), int(np.floor(np.max((0, box[1] - box[3]/2)))), int(np.ceil(np.min((width, box[0] + box[2]/2)))), int(np.ceil(np.min((height, box[1] + box[3]/2))))]
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
        
        # Video clip-level augmentation
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            bboxes = transformed['bboxes']
            labels = transformed['class_labels']
            
        image = to_tensor(Image.fromarray(image))
        target = {"boxes": torch.Tensor(bboxes), "labels": torch.tensor(labels, dtype=torch.int64)}

        return image, target, index


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

        files = [os.path.join(self.data_path,file) for file in sorted(os.listdir(self.data_path))]
        labels = [os.path.join(self.label_path,file) for file in sorted(os.listdir(self.label_path))]
        
        image_path = []
        labels_path = []
        
        for index in range(len(files)):
            video_path = files[index]
            anno_path = labels[index]
            
            lst_anno = sorted(os.listdir(anno_path))
            
            for anno in lst_anno:
                 with open(os.path.join(anno_path, anno), 'r') as f:
                    if len(f.readlines()) != 0:
                
                        image_path.append(os.path.join(video_path, anno.split(".")[0] + ".jpg"))
                        labels_path.append(os.path.join(anno_path, anno))
        
        return pd.DataFrame({
            'image_path': image_path,
            'labels_path': labels_path,
        })


class ImageDatasetBatchCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # image_batch = to_image_list(image_batch)
        
        image_batch = list(transposed_batch[0])
        targets = transposed_batch[1]
        video_ids = transposed_batch[2]
        
        return image_batch, targets, video_ids

        