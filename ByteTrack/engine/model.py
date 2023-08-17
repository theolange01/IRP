# IRP ByteTracker

import os
from tqdm import tqdm
from time import perf_counter

import torch
import numpy as np
import cv2

from ByteTrack.utils import LOGGER, colorstr, check_yaml
from ByteTrack.model.MotionDetector import MotionDetector
from ByteTrack.data import load_track_dataloader, load_validation_dataloader
from ByteTrack.utils import IterableSimpleNamespace, yaml_load
from ByteTrack.utils.Annotator import Annotator
from ByteTrack.tracker import BOTSORT, BYTETracker
from ..eval.eval import evaluate
from ..eval import CLEAR, Count, HOTA, Identity

from ultralytics import YOLO
from ultralytics.engine.results import Results

# List of available association algorithm, only BYTE has been used and tested
TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT} 


class Tracker:
    """
    Tracking algorithm. This algorithm applies a 'Tracking-by-Detection' pipeline.
    The object detector is a combination of Ultralytics Yolov8 detection model and a Background Removal Motion Detector
    The Association algorithm is a variation of the BYTE algorithm applying a filtering step to remove outlier detections.
    
    Args:
        model (str): Path the a YOLO model
    
    Attributes:
        YOLO (ultralytics.YOLO): YOLO detection model
        CLASS_NAMES_DICT (DICT[int, str]): List of the different objects type
        motion_detector (): Motion Detection algorithm
        frame_memory (List): List of the processed frame in the same frame sequence
        VideoWriter (): OpenCV Video Writer
        Annotator (): Frame Annotator object
        save_dir (str): Default save directory
        _save_dir (dir): Save Directory currently in use
        nb_frame (int): Number of frame that have already been processed
        trackers (List): List of the used trackers 

    Methods:
        __init__(): Initialise the Tracker
        reset_tracker(): Reset the memory of the tracking algorithm
        train(): Train the Yolov8 model
        __call__(): Default call for tracking
        track(): Apply tracking to the given data
        predict(): Detect objects of interests without tracking.
        validate(): Validate the tracking algorithm
        merge_detection(): Merge the lists of objects detected from the two object detectors
        write_video(): Save the annotated frame as a video
        bbox_ious(): Compute the IoU matrix between two set of bounding boxes
        register_tracker(): Initialise the tracker for tracking
        update_tracker(): Update the tracker's memory with the new detected objects
        save_txt(): Save the predicted tracking results
        save_frame(): Save the annotated frames
    """
    
    def __init__(self, model):
        """Initialise the tracking algorithm."""

        self.YOLO = YOLO(model)
        self.CLASS_NAMES_DICT = self.YOLO.names
        self.CLASS_NAMES_DICT[len(self.CLASS_NAMES_DICT)] = "other" # New class for the object to be filtered

        self.motion_detector = MotionDetector()
        self.frame_memory = []
        self.VideoWriter = None
        self.Annotator = Annotator(CLASS_NAMES_DICT=self.CLASS_NAMES_DICT)

        self.save_dir = "runs/track"
        self._save_dir = self.save_dir
        self.nb_frame = 0


    def reset_tracker(self):
        """Reset the memory of the tracking algorithm."""

        self.frame_memory = []
        self.nb_frame = 0
        if self.VideoWriter:
            self.VideoWriter.release()
            self.VideoWriter = None
        self.motion_detector.reset_memory()

        if hasattr(self, 'trackers'):
            delattr(self, 'trackers')

        if hasattr(self.YOLO.predictor, "trackers"):
                delattr(self.YOLO.predictor, "trackers")


    def train(self, data, **kwargs):
        """Train the Yolov8 model. See Ultralytics docs for information."""
        os.environ['WANDB_DISABLED'] = 'true'

        self.YOLO.train(data=data, **kwargs)
    

    def __call__(self, source, tracker="bytetrack.yaml", persist=False, save_video=True, video="tracking_results.avi", fps=None, save_txt=False, save_frame=False, use_yolo=True, use_motion_detector=True, iou_merge=0.7, verbose=False, **kwargs):
        """
        Default call for tracking.
        
        Args:
            source (): Data to use for tracking. Can be frames or directory
            tracker (str): path to the tracker configuration file
            persist (bool): Flag to reinitialise the tracking memory
            save_video (bool): whether to save the annotated frames as a video
            video (str): Name of the video. Only used if 'save_video' is True
            fps (int): FPS value to use for the video. If None and not available in the source, the results won't be saved as a video
            save_txt (bool): whether to save the tracking results in txt files
            save_frame (bool): whether to save the annotated frames
            use_yolo (bool): Whether to use the YOLO model for object detection
            use_motion_detector (bool): Whether to use the Motion Detector for object detection
            iou_merge (float): IoU threshold for detection merging between YOLO and MotionDetector predictions
            verbose (bool): Flag to display the tracking results
            kwargs: Other arguments for the YOLO model. See Ultralytics for more info
        
        Returns:
            Tracking Results
        """
        
        return self.track(source=source, tracker=tracker, persist=persist, save_video=save_video, video=video, fps=fps, save_txt=save_txt, save_frame=save_frame, use_yolo=use_yolo, use_motion_detector=use_motion_detector, iou_merge=iou_merge, verbose=verbose, **kwargs)


    def track(self, source, tracker="bytetrack.yaml", persist=False, save_video=False, video="tracking_results.avi", fps=None, save_txt=False, save_frame=False, use_yolo=True, use_motion_detector=False, iou_merge=0.7, verbose=False, **kwargs):
        """
        Run tracking algorithm on given input source.
        
        Args:
            source (): Data to use for tracking. Can be frames or directory
            tracker (str): path to the tracker configuration file
            persist (bool): Flag to reinitialise the tracking memory
            save_video (bool): whether to save the annotated frames as a video
            video (str): Name of the video. Only used if 'save_video' is True
            fps (int): FPS value to use for the video. If None and not available in the source, the results won't be saved as a video
            save_txt (bool): whether to save the tracking results in txt files
            save_frame (bool): whether to save the annotated frames
            use_yolo (bool): Whether to use the YOLO model for object detection
            use_motion_detector (bool): Whether to use the Motion Detector for object detection
            iou_merge (float): IoU threshold for detection merging between YOLO and MotionDetector predictions
            verbose (bool): Flag to display the tracking results
            kwargs: Other arguments for the YOLO model. See Ultralytics for more info
        
        Returns:
            results (List[Ultralytics.engine.results]): List of predicted objects for the input source
            frames (List[np.ndarray]): List of the frames annotated with the tracking results
        """
        
        # At least one of the two object detector should be used
        if not (use_yolo or use_motion_detector):
            raise Exception("At least one detection method should be selected.")
        
        # Load the source as a dataloader for tracking
        source = load_track_dataloader(source)
        """
        source (Dict[str])
            data (Tuple or DataLoader): Tracking Dataloader. For a single frame, it is a tuple containing a default path and the frame
            fps (int): Fps value if the input is a video or frame sequence, else None
            type (str): Whether a whole video has been passed or a single frame, "video" or "frame"
        """

        # If the source is a video, use its fps as the fps value
        if source['fps']: fps = source['fps']

        if source['type'] == 'video' and verbose:
            text = f'{"YOLO + MotionDetector" if (use_yolo and use_motion_detector) else "YOLO" if (use_yolo) else "MotionDetector"} + {tracker.split(".")[0].upper()} Tracker.'
            LOGGER.info(f"{colorstr(text)} Started Tracking: ")

        # Reinitialise the tracking memory and update the save directory
        if not persist:
            self.nb_frame = 0
            i = 1
            self._save_dir = self.save_dir
            while os.path.exists(self._save_dir):
                self._save_dir = self.save_dir + str(i)
                i += 1
        
        # register a new tracker if needed
        self.register_tracker(1, persist, tracker)
        kwargs.pop('save', None) # Not in Use here

        results = []

        ts = perf_counter()
        if use_motion_detector == False:
            # Only Yolo will be used for detection
            for index, (path, frame) in enumerate(source['data']):
                ts_i = perf_counter()
                result = self.YOLO(frame, verbose=False, **kwargs) # Detect objects

                result = self.update_tracker(result, [frame], 1, fps) # Apply association algorithm for tracking
                te_i = perf_counter()

                frame = self.Annotator(frame, result[0]) # Annotate frame
                self.frame_memory.append(frame) # Save the frame in memory

                # Save the results in a video
                if save_video:
                    os.makedirs(self._save_dir, exist_ok=True)
                    self.write_video([frame], persist, video, fps, self._save_dir)

                # Print results when verbose is True
                if verbose:
                    text = ""
                    for c in result[0].boxes.cls.unique():
                        n = (result[0].boxes.cls == c).sum()  # detections per class
                        text += f"{n} {self.CLASS_NAMES_DICT[int(c)]}{'s' * (n > 1)}, "
                    LOGGER.info(f"image {index+1}/{len(source['data'])} {path if path else 'image0.jpg'}: {frame.shape[0]}x{frame.shape[1]} {text}{(te_i - ts_i)/1000:.3f}ms")

                # Add the tracking results to the list of results
                results.append(result[0])

        else:
            for index, (path, frame) in enumerate(source['data']):
                ts_i = perf_counter()
                if use_yolo == False:
                    # only the MotionDetector is used
                    result = None
                    motion_result = self.motion_detector(frame, persist)

                    # Merge the results into the correct format
                    result = self.merge_detection(result, motion_result, [frame], paths = [path], iou=iou_merge)

                else:
                    # Both detection models are used
                    # Get the two lists of predictions
                    result = self.YOLO(frame, verbose=False, **kwargs)
                    motion_result = self.motion_detector(frame, persist)

                    # Merge the lists of predictions
                    result = self.merge_detection(result, motion_result, [frame], paths= [path], iou=iou_merge)
                
                # Apply the association algorithm for tracking
                result = self.update_tracker(result, [frame], 1, fps)
                te_i = perf_counter()

                # Add the tracking results to the list of results
                results.append(result[0])

                # Annotate the frame and update the memory
                frame = self.Annotator(frame, result[0])
                self.frame_memory.append(frame)
                
                # Save the results in a video
                if save_video:
                        os.makedirs(self._save_dir, exist_ok=True)
                        self.write_video([frame], persist, video, fps, self._save_dir)

                # Print results when verbose is True
                if verbose:
                    text = ""
                    for c in result[0].boxes.cls.unique():
                        n = (result[0].boxes.cls == c).sum()  # detections per class
                        text += f"{n} {self.CLASS_NAMES_DICT[int(c)]}{'s' * (n > 1)}, "
                    LOGGER.info(f"image {index+1}/{len(source['data'])} {path if path else 'image0.jpg'}: {frame.shape[0]}x{frame.shape[1]} {text}{(te_i - ts_i)/1000:.3f}ms")


        te = perf_counter()

        if source['type'] == "video" and verbose:
                LOGGER.info(f"Tracking Complete. Frame processing Rate: {len(source['data'].dataset)/(te-ts):.2f} frame/seconds")

        # Save all the tracking results in text files
        if save_txt:
            os.makedirs(self._save_dir, exist_ok=True)
            os.makedirs(self._save_dir + "/" +  'labels', exist_ok=True)
            self.save_txt(results, self._save_dir + "/" +  'labels')
            
        # Save every frames
        if save_frame:
            os.makedirs(self._save_dir, exist_ok=True)
            os.makedirs(self._save_dir + "/" + 'frames', exist_ok=True)
            self.save_frame(self.frame_memory[-len(source['data']):], self._save_dir + "/" + 'frames')
        
        if verbose and (save_frame or save_txt or save_video):
            LOGGER.info(f"Results saved at {self._save_dir}")

        self.nb_frame += len(results)

        # If an entire video has been processed, release the VideoWriter
        if source['type'] == 'video' and self.VideoWriter:
            self.VideoWriter.release()

        return results, self.frame_memory[-len(source['data']):]


    def predict(self, source, persist=False, save_txt=False, save_frame=False, use_yolo=True, use_motion_detector = True, iou_merge = 0.7, verbose=False, **kwargs):
        """
        Perform object detection only on the given input source.
        
        Args:
            source (): Data to use for tracking. Can be frames or directory
            persist (bool): Flag to reinitialise the detectors memory
            save_txt (bool): whether to save the detection results in txt files
            save_frame (bool): whether to save the annotated frames
            use_yolo (bool): Whether to use the YOLO model for object detection
            use_motion_detector (bool): Whether to use the Motion Detector for object detection
            iou_merge (float): IoU threshold for detection merging between YOLO and MotionDetector predictions
            verbose (bool): Flag to display the tracking results
            kwargs: Other arguments for the YOLO model. See Ultralytics for more info
        
        Returns:
            results (List[Ultralytics.engine.results]): List of detected objects for the input source
        """

        # At least one of the two object detector should be used
        if not (use_yolo or use_motion_detector):
            raise Exception("At least one detection method should be selected.")
        
        # Load the source as a dataloader for tracking
        source = load_track_dataloader(source)['data']
        """
        source (Dict[str])
            data (Tuple or DataLoader): Dataloader. For a single frame, it is a tuple containing a default path and the frame
            fps (int): Fps value if the input is a video or frame sequence, else None
            type (str): Whether a whole video has been passed or a single frame, "video" or "frame"
        """

        # Get a new directory to save the detection results
        save_dir = "runs/detect"
        i = 1
        _save_dir = save_dir
        while (os.path.exists(_save_dir)):
            _save_dir = save_dir + str(i)
            i += 1

        save_dir = _save_dir
        kwargs.pop("save", None) # Not in use

        # Reset the object detector memory
        if not persist:
            self.reset_tracker()
        
        # Perform object detection
        results=[]
        for index, (path, frame) in enumerate(source['data']):
            ts_i = perf_counter()
            if not use_motion_detector: # Only apply the YOLO detector
                result = self.YOLO(source, verbose=False, **kwargs)
                results.append(result[0])

            else:
                if use_yolo: # Apply both object detector
                    result = self.YOLO(source, verbose=False, **kwargs)
                    motion_result = self.motion_detector(source, False)

                    # Merge the lists of detections
                    result = self.merge_detection(results, motion_result, paths=[path], iou=iou_merge)

                else: # Only apply the MotionDetector
                    result = None
                    motion_result = self.motion_detector(source, False)

                    # Convert the detections into the correct format
                    result = self.merge_detection(results, motion_result, paths=[path], iou=iou_merge)
                
                results.append(result[0])

            te_i = perf_counter()

            # Save the detections into a text file
            if save_txt:
                    os.makedirs(save_dir, exist_ok=True)
                    os.makedirs(save_dir + "/" +  'labels', exist_ok=True)
                    self.save_txt(results, save_dir + "/" +  'labels')
            
            # Annotate and save the frame
            if save_frame:
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(save_dir + "/" +  'frames', exist_ok=True)
                frame = self.Annotator(frame, result[0], tracking=False)
                self.save_frame(frame,  save_dir + "/" + 'labels')

            # Display detection results is verbose is True
            if verbose:
                path = result[0].path
                text = ""
                for c in result[0].boxes.cls.unique():
                    n = (result[0].boxes.cls == c).sum()  # detections per class
                    text += f"{n} {self.CLASS_NAMES_DICT[int(c)]}{'s' * (n > 1)}, "
                LOGGER.info(f"image {index}/{len(result[0])} {path}: {frame.shape[0]}x{frame.shape[1]} {text}{(te_i - ts_i)/1000:.2f}ms")

        # Return the list of detection result
        return results


    def validate(self, source, tracker="bytetrack.yaml", use_yolo = True, use_motion_detector = False, iou_merge = 0.7, **kwargs):
        """
        Validate the tracking algorithm on given input source.
        
        Args:
            source (): Data to use for tracking. Can be frames or directory
            tracker (str): path to the tracker configuration file
            use_yolo (bool): Whether to use the YOLO model for object detection
            use_motion_detector (bool): Whether to use the Motion Detector for object detection
            iou_merge (float): IoU threshold for detection merging between YOLO and MotionDetector predictions
            kwargs: Other arguments for the YOLO model. See Ultralytics for more info
        
        Returns:
            results (List[Ultralytics.engine.results]): List of predicted objects for the input source
            frames (List[np.ndarray]): List of the frames annotated with the tracking results
        """
        
        # At least one of the two object detector should be used
        if not (use_yolo or use_motion_detector):
            raise Exception("At least one detection method should be selected.")

        text = f'{"YOLO + MotionDetector" if (use_yolo and use_motion_detector) else "YOLO" if (use_yolo) else "MotionDetector"} + {tracker.split(".")[0].upper()} Tracker:'
        LOGGER.info(f"{colorstr(text)} Tracker Validation")

        # Load the validation dataset
        source = load_validation_dataloader(source)
        """
        source : List(Dict[str, ]):
            - data: Validation dataloader. Each batch contain a single image, its path and the ground truth targets
            - fps: The fps of the video
        
        """

        kwargs.pop('save', None)
        kwargs.pop('save_txt', None)
        kwargs.pop('verbose', None)

        # Dictionary of metrics to be computed
        metrics = {'Count': {}, 'CLEAR': {}, 'HOTA': {}, 'IDF1': {}}
        
        speed = [] # Tracking speed
        for index, video in enumerate(tqdm(source)):
            # Process each video part of the validating dataset
            data = video['data']
            targets = []
            results = []

            # Reset previous tracker and register a new one
            self.register_tracker(data.batch_size, False, tracker)

            ts = perf_counter()
            if use_motion_detector == False: # Only apply the YOLO detector
                for _, frame, target in data:
                    targets.append(target)
                    result = self.YOLO(frame, verbose=False, **kwargs) # Detect objects

                    # Update tracker and perform association
                    result = self.update_tracker(result, [frame], data.batch_size, video['fps'])

                    results.append(result[0])
                
            else:
                self.motion_detector.reset_memory() # reset MotionDetector memory for security

                for path, frame, target in data:
                    targets.append(target)
                    if use_yolo == False: # Only apply the MotionDetector
                        result = None
                        motion_result = self.motion_detector(frame, persist=True) # Detect objects

                        # Convert the detections into the correct format
                        result = self.merge_detection(result, motion_result, [frame], paths = [path], iou=iou_merge)

                    else: # Apply both object detectors
                        result = self.YOLO(frame, verbose=False, **kwargs) # Detect objects
                        motion_result = self.motion_detector(frame, persist=True) # Detect Moving Objects

                        # Merge the lists of detections
                        result = self.merge_detection(result, motion_result, [frame], paths = [path], iou=iou_merge)
                    
                    # Update tracker and perform association
                    result = self.update_tracker(result, [frame], data.batch_size,  video['fps'])

                    results.append(result[0])

            te = perf_counter()
            speed.append(len(data.dataset)/(te-ts))

            # Evaluate the experimental performance of the tracker
            metric = evaluate(targets, results)
            
            metrics['Count'][str(index)] = metric['Count']
            metrics['CLEAR'][str(index)] = metric['Clear']
            metrics['IDF1'][str(index)] = metric['IDF1']
            metrics['HOTA'][str(index)] = metric['HOTA']

            # Reset the tracker memory for the next video
            self.reset_tracker()

        # Combine the metrics computed for each videos
        metrics['Count']['COMBINED_SEQ'] = Count().combine_sequences(metrics['Count'])
        metrics['CLEAR']['COMBINED_SEQ'] = CLEAR().combine_sequences(metrics['CLEAR'])
        metrics['IDF1']['COMBINED_SEQ'] = Identity().combine_sequences(metrics['IDF1'])
        metrics['HOTA']['COMBINED_SEQ'] = HOTA().combine_sequences(metrics['HOTA'])

        LOGGER.info(f'Mean Tracking Speed: {np.mean(speed):.3f} frames/second\n')

        # Display the results
        Count().print_table(metrics['Count'])
        CLEAR().print_table(metrics['CLEAR'])
        Identity().print_table(metrics['IDF1'])
        HOTA().print_table(metrics['HOTA'])

        HOTA().plot_single_tracker_results(metrics['HOTA']['COMBINED_SEQ'], 'runs/validate')

        # return the computed metrics
        return {'Count': metrics['Count']['COMBINED_SEQ'], 'CLEAR': metrics['CLEAR']['COMBINED_SEQ'], 'HOTA': metrics['HOTA']['COMBINED_SEQ'], 'IDF1': metrics['IDF1']['COMBINED_SEQ']}


    def merge_detection(self, results, motion_results, frames, paths=None, iou=0.7):
        """
        Merge the detections produced by the two detection models.
        
        Args:
            results (List[ultralytics.engine.results]): List of objects detected by the YOLO model
            motion_results (List[List[..., 4]]): List of moving objects detectted by the MotionDetector
            frames (List[np.array]): List of frames used for detection
            paths (List[str]): Path of each frames processed
            iou (int): IoU threshold for merging two detections

        Returns:
            results (List[ultralytics.engine.results]): Merged list of detected objects
        """

        # If only YOLO if used, no need for merging
        if not motion_results:
            return results

        # If only the MotionDetector has been used, the format of the detections needs to be changed
        elif not results:
            results = []
            for i in range(len(motion_results)):
                preds = np.array(motion_results[i])
                if len(preds):
                    preds = torch.tensor(np.hstack((preds, np.array([[0.8, len(self.CLASS_NAMES_DICT) - 1]]*len(preds)))))
                    # Should be of shape [nb_box, 6] with x, y, x, y, conf, label_id
                else:
                    preds = torch.empty(0,6)

                if paths:
                    path = paths[i]
                else: 
                    path = "image0.jpg"
                
                # Change the format of the detections results
                result = Results(frames[i], path=path, names=self.CLASS_NAMES_DICT, boxes=preds)
                results.append(result)

        # When both detection models have been used, the detected objects need to be merged
        else:
            device = results[0].boxes.xyxy.device
            for index in range(len(results)):
                if len(motion_results[index]):
                    box1 = results[index].boxes.xyxy.to(torch.device('cpu')).numpy()
                    box2 = np.array(motion_results[index])

                    iou_matrix = self.bbox_ious(box1, box2) # Compute the IoU between the detected objects of the two detection lists

                    # Get the matching index given the IoU threshold
                    matched_detections_index = np.any(iou_matrix>iou, axis=0)
                    remaining_detections_index = np.logical_not(matched_detections_index)

                    box2 = box2[remaining_detections_index].reshape(-1, 4)
                    boxes = np.hstack((box2, np.array([[0.8, len(self.CLASS_NAMES_DICT)-1]]*len(box2))))

                    # Update the detection results
                    remaining_detections = torch.tensor(boxes, device = device)
                    results[index].update(boxes=torch.cat((results[0].boxes.data, remaining_detections)))

        return results


    def write_video(self, frames, persist, video, fps, save_dir):
        """
        Write the video with the annotated frames after tracking.
        
        Args:
            frames (List[np.array]): List of frames to write in the video
            persist (bool): Whether to keep the previous videoWriter or initialise a new one
            video (str): Name of the video file
            fps (int): FPS value used to write the video
            save_dir (str): Directory where the video should be saved
        """

        if persist: # Keep the same VideoWriter if possible
            if not self.VideoWriter:
                if fps:
                    # Initialise a new VideoWriter
                    height, width = frames[0].shape[:2]
                    VIDEO_CODEC = "MJPG"
                    self.VideoWriter = cv2.VideoWriter(os.path.join(save_dir, video), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))

                else:
                    # Without fps, the VideoWriter cannot be initialised
                    LOGGER.warning("WARNING ⚠️: FPS not given, cannot save the tracking video result")
                    return None

            # Write the video
            for frame in frames:
                self.VideoWriter.write(frame)

        else: # The VideoWriter needs to be reset
            if fps:
                # Initialise a new VideoWriter
                height, width = frames[0].shape[:2]
                VIDEO_CODEC = "MJPG"
                self.VideoWriter = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))
            else:
                # Without fps, the VideoWriter cannot be initialised
                LOGGER.warning("WARNING ⚠️: FPS not given, cannot save the tracking video result")
                self.VideoWriter = None
                return None

            # Write the video
            for frame in frames:
                if self.VideoWriter:
                    self.VideoWriter.write(frame)


    def bbox_ious(self, box1, box2, eps=1e-7):
        """
        Calculate the Intersection over Union (IoU) between pairs of bounding boxes.

        Args:
            box1 (np.array): A numpy array of shape (n, 4) representing 'n' bounding boxes.
                             Each row is in the format (x1, y1, x2, y2).
            box2 (np.array): A numpy array of shape (m, 4) representing 'm' bounding boxes.
                             Each row is in the format (x1, y1, x2, y2).
            eps (float, optional): A small constant to prevent division by zero. Defaults to 1e-7.

        Returns:
            (np.array): A numpy array of shape (n, m) representing the IoU scores for each pair
                        of bounding boxes from box1 and box2.

        Note:
            The bounding box coordinates are expected to be in the format (x1, y1, x2, y2).
        """

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        # Intersection area
        inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                    (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

        # boxes area
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        return inter_area / (box2_area + box1_area[:, None] - inter_area + eps)


    def register_tracker(self, batch_size, persist, tracker):
        """
        Register a new tracker.
        
        Args:
            batch_size (int): Number of frame in a batch. Usually batch size is 1
            persist (bool): whether to keep the same tracker or register a new one
            tracker (str): Path to the trackers characteristics
        """

        # If no trackers were already registered
        if not hasattr(self, 'trackers'):
            self.reset_tracker() # reset the memory of every component
            tracker = check_yaml(tracker)
            cfg = IterableSimpleNamespace(**yaml_load(tracker)) # Read and load the tracker characteristics
            assert cfg.tracker_type in ['bytetrack', 'botsort'], \
                f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
            trackers = []
            for _ in range(batch_size):
                tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
                trackers.append(tracker)
            
            self.trackers = trackers

        else:
            if persist == False: # The tracker need to be reinitialised
                self.reset_tracker()
                
                tracker = check_yaml(tracker)
                cfg = IterableSimpleNamespace(**yaml_load(tracker)) # Read and load the tracker characteristics
                assert cfg.tracker_type in ['bytetrack', 'botsort'], \
                    f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
                trackers = []
                for _ in range(batch_size):
                    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
                    trackers.append(tracker)
                
                self.trackers = trackers
    

    def update_tracker(self, results, frames, batch_size, fps=None):
        """
        Update the tracker memory and perform data association and filtering.
        
        Args:
            results (List[ultralytics.engine.results]): Detection Results
            frames (List[np.array]): List of frames processed
            batch_size (int): Number of frame in a batch. Usually batch size is 1
            fps (int): FPS value used for the use of velocity for filtering

        Returns:
            results (List[ultralytics.engine.results]): List of tracking results
        """

        for i in range(batch_size):
            det = results[i].boxes.to(torch.device('cpu')).numpy() # Get the detected objects location
            if len(det) == 0:
                continue
            tracks = self.trackers[i].update(det, frames[i], len(self.CLASS_NAMES_DICT)-1, fps) # Update tracker
            if len(tracks) == 0:
                results[i].update(boxes=torch.empty(0,6))
                continue
            idx = tracks[:, -1].astype(int)
            results[i] = results[i][idx]
            results[i].update(boxes=torch.as_tensor(tracks[:, :-1])) # Update the results to include the tracking ID
        
        return results

    def save_txt(self, results, txt_file):
        """
        Save the results in text files.
        
        Args:
            results (List[ultralytics.engine.results]): List of tracking results
            txt_file (str): Directory where the label files should be saved
        """

        for index, result in enumerate(results):
            texts = []
            boxes = result.boxes
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item()) # Class, confidence score and tracking ID
                line = (c, *d.xywhn.view(-1)) # Bounding boxes in YOLO format
                line += (conf, ) * True + (() if id is None else (id, ))
                texts.append(('%g ' * len(line)).rstrip() % line)

            with open(os.path.join(txt_file, f"{'0'*(6-len(str(self.nb_frame+index+1))) + str(self.nb_frame+index+1)}.txt"), 'a') as f:
                f.writelines(text + '\n' for text in texts)
    
    def save_frame(self, frames, frame_dir):
        """
        Save the frames.
        
        Args:
            frames (List[np.array]): List of frames to be saved
            frame_dri (str): Directory where the frames should be saved
        """

        for index, frame in enumerate(frames):
            cv2.imwrite(os.path.join(frame_dir, f"{'0'*(6-len(str(self.nb_frame+index+1))) + str(self.nb_frame+index+1)}.jpg"), frame)