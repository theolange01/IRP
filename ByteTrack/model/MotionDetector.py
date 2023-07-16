import os
from glob import glob

import numpy as np
import cv2
from PIL import Image

from ..utils import LOGGER

import torch
from torchvision.transforms.functional import to_pil_image

class MotionDetector:

    def __init__(self, threshold=60):

        self.thresh = threshold

        self.memory = None
        self.dim = None
        
        self.backsub= cv2.createBackgroundSubtractorMOG2()

    def reset_memory(self):
        self.memory = None
        self.dim = None
        self.backsub= cv2.createBackgroundSubtractorMOG2()

    def __call__(self, source=None, persist=False):
        return self.detect(source, persist=persist)
    
    def detect(self, source="", persist=False):
        if not source:
            return []
        
        if isinstance(source, str):
            if not os.path.exists(source):
                LOGGER.warning("⚠️ Warning: File not Found")
                return []
            
        if not persist:
            self.reset_memory()
        
        lst_frames = self.get_frames(source)

        proposals = []
        if len(lst_frames) == 0:
            return [[]] # No frames available or readable, so no proposals
        

        if len(lst_frames) == 1:
            if not self.memory:
                # If not frames are stored in memory, the algorithm can't find the moving objects
                # Save the frame in memory and return an empty list
                self.memory = lst_frames
                self.height, self.width  = self.memory.shape[:2]

                dim = (self.width*2, self.height*2)

                frame = cv2.resize(self.memory, dim, interpolation = cv2.INTER_AREA)

                _ = self.backsub.apply(frame)
                return [[]]

        else:
            if not self.memory:
                # When a list of frames is given, use the first one to initialise the detector
                self.memory = lst_frames
                self.height, self.width  = self.memory.shape[:2]

                dim = (self.width*2, self.height*2)

                frame = cv2.resize(self.memory, dim, interpolation = cv2.INTER_AREA)

                _ = self.backsub.apply(frame)

                lst_frames = lst_frames[1:]
                proposals.append([]) # No proposals for the first frame in this case

            
        for frame in lst_frames:
            dim = (self.width*2, self.height*2)

            curr_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            curr_frame_gray = cv2.GaussianBlur(curr_frame_gray, (5, 5), cv2.BORDER_DEFAULT)

            fgmask = self.backsub.apply(curr_frame_gray)

            kernel = np.ones((10, 10), np.uint8)
            diff_frame = cv2.dilate(fgmask, kernel, 2)

            thresh_frame = cv2.threshold(diff_frame, self.thresh, 255, cv2.THRESH_BINARY)[1]
            
            erode_frame = cv2.erode(thresh_frame, kernel, 3)

            contours, _ = cv2.findContours(image=erode_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            proposal = []
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                proposal.append([x/2, y/2, x/2+w/2, y/2+h/2])

            proposals.append(proposal)
        
        self.memory = lst_frames[-1]

        return proposals


    def get_frames(self, source):

        if isinstance(source, str):
            if source.endswith((".avi", '.mov', '.mp4')):
                return self.read_video(source)
            elif source.endswith(('.jpg', '.jpeg', '.png', '.PNG')):
                return [cv2.imread(source)]
            else:
                return self.read_frames(source)
            
        elif isinstance(source, (list, tuple)):
            if isinstance(source[0], np.array):
                return source
            elif isinstance(source[0], Image):
                return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in source]
            elif isinstance(source[0], torch.Tensor):
                return [cv2.cvtColor(np.array(to_pil_image(image.to(torch.device('cpu')))), cv2.COLOR_RGB2BGR) for image in source]
            else:
                raise NotImplementedError(f"The type of the frames {source[0].type} is not handled!")
        

        elif isinstance(source, np.ndarray):
            try:
                source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            except:
                pass

            return [source]

        elif isinstance(source, Image):
            return [cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)]
        
        elif isinstance(source, torch.Tensor):
            return [cv2.cvtColor(np.array(to_pil_image(source.to(torch.device('cpu')))), cv2.COLOR_RGB2BGR)]
                    
        else:
            raise NotImplementedError(f"The source type {source.type} is not handled!")
        
    def read_video(self, source):
        lst_frames = []

        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lst_frames.append(frame)
        
        cap.release()

        return lst_frames
    
    def read_frames(self, source):
        lst_frames = []

        file_type = ["*.jpg", "*.jpeg", "*.png", "*.PNG"]
        index = 0
        lst_frame = []

        while len(lst_frame) == 0:
            lst_frame = sorted(glob(os.path.join(source, file_type[index])))
            index += 1

            if index == 4:
                break

        if len(lst_frame) == 0: return []
        
        for frame_path in lst_frame:
            frame = cv2.imread(frame_path)              

            lst_frames.append(frame)

        return lst_frames