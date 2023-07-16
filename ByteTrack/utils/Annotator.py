from typing import Dict, Optional

import numpy as np 
import torch
from PIL import Image

import cv2
import supervision as sv

from torchvision.transforms.functional import to_pil_image

class Annotator:

    def __init__(self,thickness: int = 2,text_scale: float = 0.5,text_thickness: int = 1, text_padding: int = 2, CLASS_NAMES_DICT: Dict[int, str] = {}):

        self.box_annotator = sv.BoxAnnotator(thickness=thickness, text_scale=text_scale,
                                             text_thickness=text_thickness, text_padding=text_padding)
        
        self.CLASS_NAMES_DICT = CLASS_NAMES_DICT

    def __call__(self, frame, detections):
        frame = self._check_frame(frame)
        return self.annotate_frame(frame, detections)
    
    def annotate_frame(self, frame, result):
        class_id = result.boxes.cls.cpu().numpy().astype(int)

        if len(class_id) == 0:
            return frame
       
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int)
        )
        
        try:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        except:
            detections.tracker_id = None
        

        labels = [f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, tracker_id in detections]   

        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame

    
    def _check_frame(self, frame):
        if isinstance(frame, np.ndarray):
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except:
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                except:
                    pass
        
        elif isinstance(frame, Image):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        elif isinstance(frame, torch.Tensor):
            frame = cv2.cvtColor(np.array(to_pil_image(frame.to(torch.device("cpu")))), cv2.COLOR_RGB2BGR)

        else:
            raise NotImplementedError(f"The type of the given frame {type(frame)} cannot be handled")

        return frame
