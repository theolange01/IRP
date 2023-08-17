# IRP ByteTracker

from typing import Dict

import numpy as np 
import torch
from PIL import Image
import cv2
import supervision as sv

from torchvision.transforms.functional import to_pil_image


class Annotator:
    """
    Annotator class. Locate detection or tracking results on the frames.

    Args:
        thickness (int): The thickness of the bounding box lines, default is 2
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box, default is 1
        text_padding (int): The padding around the text on the bounding box, default is 2
        CLASS_NAMES_DICT (Dict[int, str]): List of the different objects type

    Attributes:
        box_annotator (sv.BoxAnnotator):
        CLASS_NAMES_DICT (Dict[int, str]): List of the different objects type

    Methods:
        __init__(): Initialise the frame Annotator object
        __call__(): Default call function for frame annotation
        annotate_frame(): Annotate the given frame with the given tracking or detection result
        _check_frame(): Verify the input frame    
    """

    def __init__(self, thickness: int = 2, text_scale: float = 0.5, text_thickness: int = 1, text_padding: int = 2, CLASS_NAMES_DICT: Dict[int, str] = {}):
        """
        Initialise the frame annotator.
        
        Args:
            thickness (int): The thickness of the bounding box lines, default is 2
            text_scale (float): The scale of the text on the bounding box, default is 0.5
            text_thickness (int): The thickness of the text on the bounding box, default is 1
            text_padding (int): The padding around the text on the bounding box, default is 2
            CLASS_NAMES_DICT (Dict[int, str]): List of the different objects type
        """

        # Create the BoxAnnotator object
        self.box_annotator = sv.BoxAnnotator(thickness=thickness, text_scale=text_scale,
                                             text_thickness=text_thickness, text_padding=text_padding)
        
        # Save the different class names
        self.CLASS_NAMES_DICT = CLASS_NAMES_DICT

    def __call__(self, frame, detections, tracking=True):
        """
        Default call function to annotate a frame.
        
        Args:
            frame (np.array): the frame to annotated
            detections (ultralytics.engine.results): Tracking or detection results for the frame
            tracking (bool): Whether the results correspond to tracking results or not

        Returns: 
            np.array: The annotated frame
        """
        
        # Verify the input frame and annotate it with the given results
        frame = self._check_frame(frame)
        return self.annotate_frame(frame, detections, tracking=tracking)
    
    def annotate_frame(self, frame, result, tracking=True):
        """
        Annotate a Frame with the input tracking or detection results.
        
        Args:
            frame (np.array): the frame to annotated
            detections (ultralytics.engine.results): Tracking or detection results for the frame
            tracking (bool): Whether the results correspond to tracking results or not

        Returns: 
            frame (np.array): The annotated frame
        """
        
        # Get the List of detected classes
        class_id = result.boxes.cls.cpu().numpy().astype(int)

        # If there are no results, return the frame
        if len(class_id) == 0:
            return frame
       
        # Create the supervision Detections object
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int)
        )
        
        # Add the tracking ID
        try:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        except:
            detections.tracker_id = None
        
        if tracking:
            # If the results correspond to tracking results, add the tracking ID to the annotations
            labels = [f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, tracker_id in detections]   
        else:
            # Detection results
            labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]   

        # Annotate the frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame

    
    def _check_frame(self, frame):
        """
        Verify the input frame.
        
        Args:
            frame (np.array): Frame to annotated

        Returns:
            frame (np.array): Frame to annotated
        """
        
        if isinstance(frame, np.ndarray):
            # Frame in OpenCV BGR format
            pass
        
        elif isinstance(frame, Image.Image):
            # Convert the frame to OpenCV frame and BGR format
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        elif isinstance(frame, torch.Tensor):
            # Convert the frame to OpenCV frame and BGR format
            frame = cv2.cvtColor(np.array(to_pil_image(frame.to(torch.device("cpu")))), cv2.COLOR_RGB2BGR)

        else:
            raise NotImplementedError(f"The type of the given frame {type(frame)} cannot be handled")

        return frame
