# IRP ByteTracker

import os
from glob import glob

import numpy as np
import cv2
from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image

from ..utils import LOGGER


class MotionDetector:
    """
    Background Removal Motion Detector.
    Applies Background Extraction and frame processing techniques to detect moving objects from a sequence of frames.

    Args:
        threshold (int): Value used for the Threshold operation

    Attributes:
        thresh (int): threshold value used for the Threshold operation
        memory (List[np.array]): List of frames previousy processed
        dim (Tuple(int, int)): Dimension of the Frames after resizing
        backsub (): Background Subtraction algorithm

    Methods:
        __init__(): Initialises the Motion Detector
        reset_memory(): Reset the Detector memory
        __call__(): Default call for detection
        detect(): Perform object detection on the given input
        get_frames(): Check the input data and load the frames
        read_video(): Read the frames from the video
        read_frames(): Load the frames from a directory
    """

    def __init__(self, threshold=60):
        """Initialise the Motion Detector."""

        self.thresh = threshold
        self.memory = None
        self.dim = None
        
        # Initialise the background extraction algorithm
        self.backsub= cv2.createBackgroundSubtractorMOG2()


    def reset_memory(self):
        """Reset the background extraction algorithm."""

        self.memory = None
        self.dim = None
        self.backsub= cv2.createBackgroundSubtractorMOG2()


    def __call__(self, source=None, persist=False):
        """
        Call the detection method.

        Args:
            source (str, np.array, PIL.Image, tensor, list, or tuple): Frame, frame sequence or video to use for prediction
            persist (bool): Flag use to reinitialise the detector for the processing of a new video

        Returns:
            detections (List[List[int, 4]]): List of detected moving objects for each frame in the source        
        """

        return self.detect(source, persist=persist)
    

    def detect(self, source="", persist=False):
        """
        Perform object detection on the given input source

        Args:
            source (str, np.array, PIL.Image, tensor, list, or tuple): Frame, frame sequence or video to use for prediction
            persist (bool): Flag use to reinitialise the detector for the processing of a new video

        Returns:
            detections (List[List[int, 4]]): List of detected moving objects for each frame in the source
        """ 

        # If wrong path given, print a warning and returns an empty detection 
        if isinstance(source, str):
            if not os.path.exists(source):
                LOGGER.warning("⚠️ Warning: File not Found")
                return []
            
        # Reinitialise the memory for a new video or frame sequence
        if not persist:
            self.reset_memory()
        
        # Load the different frames
        lst_frames = self.get_frames(source)

        detections = []
        if len(lst_frames) == 0:
            return [[]] # No frames available or readable, so no detections
        

        if len(lst_frames) == 1:
            if not self.memory:
                # If no frames are stored in memory, the algorithm can't find the moving objects as it needs to be initialised
                # Save the frame in memory and return an empty list
                self.memory = lst_frames
                self.height, self.width  = lst_frames[0].shape[:2]

                dim = (self.width*2, self.height*2)

                # Transform the frame: increase the resolution, convert to gray and apply Gaussian Blur
                frame = cv2.resize(lst_frames[0], dim, interpolation = cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

                # Initialise the background extraction algorithm
                _ = self.backsub.apply(frame)
                return [[]]

        else:
            if not self.memory:
                # When a list of frames is given, use the first one to initialise the detector if the memory is empty
                self.memory = [lst_frames[0]]
                self.height, self.width  = lst_frames[0].shape[:2]

                dim = (self.width*2, self.height*2)

                # Transform the frame: increase the resolution, convert to gray and apply Gaussian Blur
                frame = cv2.resize(self.memory, dim, interpolation = cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)

                # Initialise the background extraction algorithm
                _ = self.backsub.apply(frame)

                lst_frames = lst_frames[1:]
                detections.append([]) # No detections for the first frame

            
        for frame in lst_frames:
            dim = (self.width*2, self.height*2)

            # Transform the frame: increase the resolution, convert to gray and apply Gaussian Blur
            curr_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_frame_gray = cv2.GaussianBlur(curr_frame_gray, (5, 5), cv2.BORDER_DEFAULT)

            # Extract a mask of the foreground
            fgmask = self.backsub.apply(curr_frame_gray)

            # Dilate the foreground moving objects
            kernel = np.ones((5, 5), np.uint8)
            diff_frame = cv2.dilate(fgmask, kernel, 2)

            # Apply threshold transformation
            thresh_frame = cv2.threshold(diff_frame, self.thresh, 255, cv2.THRESH_BINARY)[1]
            
            # Erode the foregound objects
            erode_frame = cv2.erode(thresh_frame, kernel, 3)

            # Obtain the contour of each detected moving objects
            contours, _ = cv2.findContours(image=erode_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            detection = []
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour) # Get the location of the bounding boxes
                detection.append([x/2, y/2, x/2+w/2, y/2+h/2])

            detections.append(detection)
        
        # Update the frame memory
        self.memory.extend(lst_frames)

        return detections


    def get_frames(self, source):
        """
        Check the source and load the frames.
        
        Args:
            source: The data source. Can be a frame, a path to a single frame or video, or a path to a directory containing multiple frames
        
        Returns:
            List[np.array]: List of frames to be processed in BGR format
        """

        # When the source is in str format
        if isinstance(source, str):
            if source.endswith((".avi", '.mov', '.mp4')): # Video
                return self.read_video(source)
            elif source.endswith(('.jpg', '.jpeg', '.png', '.PNG')): # Frame
                return [cv2.imread(source)] # BGR
            else:
                return self.read_frames(source) # Read the frames from the given directory
            
        # List or Tuple of frames already loaded
        elif isinstance(source, (list, tuple)):
            if isinstance(source[0], np.ndarray):
                return source # Frame in BGR format
            elif isinstance(source[0], Image.Image):
                return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in source] # Convert to BGR format
            elif isinstance(source[0], torch.Tensor):
                return [cv2.cvtColor(np.array(to_pil_image(image.to(torch.device('cpu')))), cv2.COLOR_RGB2BGR) for image in source] # Convert to BGR format
            else:
                raise NotImplementedError(f"The type of the frames {source[0].type} is not handled!")
        
        # The source is a single frame
        elif isinstance(source, np.ndarray):
            # The source is a single frame, it should be in BGR format
            return [source]

        elif isinstance(source, Image.Image):
            # The source is a single frame, it is converted to BGR format
            return [cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)]
        
        elif isinstance(source, torch.Tensor):
            # The source is a single frame, it is converted to BGR format
            return [cv2.cvtColor(np.array(to_pil_image(source.to(torch.device('cpu')))), cv2.COLOR_RGB2BGR)]
                    
        else:
            raise NotImplementedError(f"The source type {source.type} is not handled!")
        

    def read_video(self, source):
        """
        Read the frames from a video.
        
        Args:
            source (str): Path to a video

        Returns:
            lst_frames (List[np.array]): List of frames in the input video
        """
        
        lst_frames = []

        cap = cv2.VideoCapture(source)

        # Read and extract each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lst_frames.append(frame)
        
        cap.release()

        return lst_frames
    
    def read_frames(self, source):
        """
        Load frames from a folder containing a frame sequence.
        
        Args:
            source (str): Path to a directory containing one or multiple frames

        Returns:
            lst_frames (List[np.array]): List of the loaded frames
        """
        
        lst_frames = []

        file_type = ["*.jpg", "*.jpeg", "*.png", "*.PNG"]
        index = 0
        lst_frame = []

        # Every frames should have the same format.
        while len(lst_frame) == 0:
            lst_frame = sorted(glob(os.path.join(source, file_type[index])))
            index += 1

            if index == 4:
                break
        
        # If no frames are found, return an empty list
        if len(lst_frame) == 0: return []
        
        # Load every frames
        for frame_path in lst_frame:
            frame = cv2.imread(frame_path)              

            lst_frames.append(frame)

        return lst_frames