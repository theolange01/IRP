import os
from pathlib import Path
from typing import Union
from tqdm import tqdm
from time import perf_counter

import torch
from torch import nn
import numpy as np
import cv2

from ByteTrack.utils import LOGGER, colorstr, check_yaml
from ByteTrack.model.MotionDetector import MotionDetector
from ByteTrack.data import load_track_dataset, load_validation_dataset
from ByteTrack.utils import Annotator, IterableSimpleNamespace, yaml_load
from ByteTrack.tracker import BOTSORT, BYTETracker
from ..eval.eval import evaluate
from ..eval import CLEAR, Count, HOTA

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

class Tracker:
    def __init__(self, model, imgsz=(1920, )):
        self.YOLO = YOLO(model)
        self.CLASS_NAMES_DICT = self.YOLO.names
        self.CLASS_NAMES_DICT[len(self.CLASS_NAMES_DICT)] = "other"

        self.motion_detector = MotionDetector()
        self.frame_memory = []
        self.VideoWriter = None
        self.Annotator = Annotator(CLASS_NAMES_DICT=self.CLASS_NAMES_DICT)

        self.save_dir = "runs/track"
        self.nb_frame = 0

    def reset_tracker(self):
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
        os.environ['WANDB_DISABLED'] = 'true'

        self.YOLO.train(data=data, **kwargs)
    
    def __call__(self, source, tracker, persist=False, save_video=True, video="tracking_results.avi", fps=25, save_text=False, save_frame=False, use_yolo=True, use_motion_detector=False, iou_merge=0.7, **kwargs):
        return self.track(source=source, tracker=tracker, persist=persist, save_video=save_video, video=video, fps=fps, save_text=save_text, save_frame=save_frame, use_yolo=use_yolo, use_motion_detector=use_motion_detector, iou_merge=iou_merge, **kwargs)

    def track(self, source, tracker, persist=False, save_video=False, video="tracking_results.avi", fps=25, save_text=False, save_frame=False, use_yolo=True, use_motion_detector=False, iou_merge=0.7, **kwargs):
        if not (use_yolo or use_motion_detector):
            raise Exception("At least one detection method should be selected.")
        
        source = load_track_dataset(source)
        """
        Dict[str]
        data: dataloader (shuffle=False): path of img "image0.jpg" by default, frame
        fps  
        type: "video" or "frame" -> Whether a whole video has been passed or a single frame
        
        """

        if source['fps']: fps = source['fps']

        if source['type'] == 'video':
            text = f'{"YOLO + MotionDetector" if (use_yolo and use_motion_detector) else "YOLO" if (use_yolo) else "MotionDetector"} + {tracker.split(".")[0].upper()} Tracker:'
            LOGGER.info(f"{colorstr(text)} Started Tracking: ")

        results = []
        if not persist:
            self.nb_frame = 0
            i = 1
            save_dir = self.save_dir
            while os.path.exists(save_dir):
                save_dir = self.save_dir + str(i)
                i += 1
        
        self.register_tracker(source['data'], persist, tracker)
        verbose = 'verbose' in kwargs.keys()
                
        ts = perf_counter()
        if use_motion_detector == False:
            ts = perf_counter()
            for _, frame in source['data']:
                ts_i = perf_counter()
                result = self.YOLO(frame, save_txt=False, save=False, verbose=False, **kwargs)

                self.update_tracker(self, result, [frame], source['data'].batch_size)
                te_i = perf_counter()

                frame = self.Annotator(frame, result[0])
                self.frame_memory.append(frame)

                if save_video:
                    os.makedirs(save_dir, exist_ok=True)
                    self.write_video(frame, result, persist, video, fps, save_dir)

                if verbose:
                    text = ""
                    for c in result[0].boxes.cls.unique():
                        n = (result[0].boxes.cls == c).sum()  # detections per class
                        text += f"{n} {self.CLASS_NAMES_DICT[int(c)]}{'s' * (n > 1)}, "
                    LOGGER.info(f"image {index}/{len(source['data'])} {path}: {frame.shape[0]}x{frame.shape[1]} {text}{(te_i - ts_i)/1000:.1f}ms")


                results.append(result[0])

        else:
            for index, (path, frame) in enumerate(source['data']):
                ts_i = perf_counter()
                if use_yolo == False:
                    result = None
                    motion_result = self.motion_detector(frame, persist)

                    result = self.merge_detection(result, motion_result, [frame], paths = [path], iou=iou_merge)

                else:
                    result = self.YOLO(frame, verbose=False, **kwargs)
                    motion_result = self.motion_detector(frame, persist)

                    result = self.merge_detection(result, motion_result, [frame], paths= [path], iou=iou_merge)
                
                self.update_tracker(self, result, [frame], source['data'].batch_size)
                te_i = perf_counter()

                results.append(result[0])

                frame = self.Annotator(frame, result[0])
                self.frame_memory.append(frame)
                

                if save_video:
                        os.makedirs(save_dir, exist_ok=True)
                        self.write_video(frame, result, persist, video, fps, save_dir)

                if verbose:
                    text = ""
                    for c in result[0].boxes.cls.unique():
                        n = (result[0].boxes.cls == c).sum()  # detections per class
                        text += f"{n} {self.CLASS_NAMES_DICT[int(c)]}{'s' * (n > 1)}, "
                    LOGGER.info(f"image {index}/{len(source['data'])} {path}: {frame.shape[0]}x{frame.shape[1]} {text}{(te_i - ts_i)/1000:.1f}ms")


        te = perf_counter()

        if source['type'] == "video":
                LOGGER.info(f"Tracking Complete. Frame processing Rate: {int(len(source['data'].dataset)/(te-ts))} frame/seconds")

        if save_text:
                os.makedirs(save_dir, exist_ok=True)
                self.save_txt(results, os.path.join(save_dir, 'labels'))
            
        if save_frame:
            os.makedirs(save_dir, exist_ok=True)
            self.save_frame(self.frame_memory[-len(source['data']):], os.path.join(save_dir, 'frames'))

        self.nb_frame += len(results)

        return results, self.frame_memory[-len(source['data']):]

    def predict(self, source, save_text=False, save_frame=False, use_yolo=True, use_motion_detector = False, use_feat_filtering = False, iou_merge = 0.7, **kwargs):
        if not (use_yolo or use_motion_detector):
            raise Exception("At least one detection method should be selected.")
        
        save_dir = "runs/detect"
        i = 1
        _save_dir = save_dir
        while (os.path.exists(_save_dir)):
            _save_dir = save_dir + str(i)
            i += 1

        save_dir = _save_dir
        
        results=None
        if not use_motion_detector:
            results = self.YOLO(source, save=False, save_txt=False, **kwargs)

            return results

        motion_results=None
        if use_yolo:
            results = self.YOLO(source, verbose=False, save=False, save_txt=False, **kwargs)
            if use_feat_filtering:
                motion_results = self.motion_detector(source, [result.boxes.xyxy.to(torch.device('cpu')).numpy() for result in results])
            else:
                motion_results = self.motion_detector(source, False)

        else:
                motion_results = self.motion_detector(source, False)

        results = self.merge_detection(results, motion_results, source, iou=iou_merge)

        """
        if save_text:
                os.makedirs(save_dir, exist_ok=True)
                self.save_txt(results, os.path.join(save_dir, 'labels'))
        
        if save_frame:
            os.makedirs(save_dir, exist_ok=True)
            self.save_frame(self.frame_memory[-len(source['data']):], os.path.join(save_dir, 'frames'))

        
        verbose = 'verbose' in kwargs.keys()
        if verbose:
            for index, result in enumerate(results):
                path = result.path
                text = ""
                for c in result.boxes.cls.unique():
                    n = (result.boxes.cls == c).sum()  # detections per class
                    text += f"{n} {self.CLASS_NAMES_DICT[int(c)]}{'s' * (n > 1)}, "
                LOGGER.info(f"image {index}/{len(results)} {path}: {frame.shape[0]}x{frame.shape[1]} {text}{(te_i - ts_i)/1000:.1f}ms")
        """
        # LOGGER info for detection in each frame as YOLO

        return results

    def validate(self, source, tracker, use_yolo = True, use_motion_detector = False, iou_merge = 0.7, **kwargs):
        if not (use_yolo or use_motion_detector):
            raise Exception("At least one detection method should be selected.")

        text = f'{"YOLO + MotionDetector" if (use_yolo and use_motion_detector) else "YOLO" if (use_yolo) else "MotionDetector"} + {tracker.split(".")[0].upper()} Tracker:'
        LOGGER.info(f"{colorstr(text)} Tracker Validation")

        source = load_validation_dataset(source)
        """
        source : List(Dict[str, ]):
            - dataloader (batch_size=1, shuffle=False): Give all the images for inference in the right order and the targets, image_path
            - fps : The fps of the video
        
        """

        total_metrics = []
        speed = []
        for video in tqdm(source):
            data = video['data']
            targets = []
            results = []

            if hasattr(self.YOLO.predictor, "trackers"):
                delattr(self.YOLO.predictor, "trackers")

            ts = perf_counter()
            

            if use_motion_detector == False:
                for _, frame, target in data:
                    targets.append(target)
                    result = self.YOLO.track(frame, tracker, persist = True, verbose=False, **kwargs)[0]

                    results.append(result)
                
            else:
                
                self.register_tracker(data, False, tracker)

                paths = []

                for path, frame, target in data:
                    paths.append(path)
                    if use_yolo == False:
                        result = None
                        motion_result = self.motion_detector(frame)

                        result = self.merge_detection(result, motion_result, [frame], path= [path], iou=iou_merge)

                    else:
                        result = self.YOLO(frame, verbose=False, **kwargs)
                        motion_result = self.motion_detector(frame)

                        result = self.merge_detection(result, motion_result, [frame], path = [path], iou=iou_merge)
                    
                    self.update_tracker(self, result, [frame], data.batch_size)

                    results.append(result[0])

            te = perf_counter()
            speed.append(int(len(data.dataset)/(te-ts)))
            
            metrics = evaluate(targets, results)
            total_metrics.append(metrics)

            self.reset_tracker()

        print(total_metrics[0])

        count = Count().combine_sequences([metric['Count'] for metric in total_metrics])
        clear = CLEAR().combine_sequences([metric['CLEAR'] for metric in total_metrics])
        hota = HOTA().combine_sequences([metric['HOTA'] for metric in total_metrics])

        LOGGER.info(f'Mean Tracking Speed: {np.mean(speed)}\n')

        Count().print_table({'COMBINES_SEQS': count})
        CLEAR().print_table({'COMBINES_SEQS': clear})
        HOTA().print_table({'COMBINES_SEQS': hota})

        HOTA().plot_single_tracker_results(hota, 'runs/validate')

        print(count)
        print(clear)
        print(hota)
        print(total_metrics[0])

        # display metrics first on LOGGER
        # Save graph and other in image format or csv
        return total_metrics


    def merge_detection(self, results, motion_results, frames, paths=None, iou=0.7):
        if not motion_results:
            pass

        elif not results:
            results = []
            for i in range(len(motion_results)):
                preds = np.array(motion_results[i])
                preds = torch.tensor(np.hstack((preds, np.array([0.8, len(self.CLASS_NAMES_DICT) - 1]))))
                # Should be of shape [nb_box, 6] with x, y, x, y, conf, label_id

                if paths:
                    path = paths[i]
                else: 
                    path = "image0.jpg"
                

                result = Results(frames[i], path=path, names=self.CLASS_NAMES_DICT, boxes=preds)
                results.append(result)

        else:
            device = results[0].boxes.xyxy.device
            for index in range(len(results)):
                height, width = results[index].boxes.orig_shape
                box1 = results[index].boxes.xyxy.to(torch.device('cpu')).numpy()
                box2 = motion_results[index]

                iou_matrix = self.bbox_ious(box1, box2)

                matched_detections_index = np.any(iou_matrix>iou, axis=1)
                remaining_detections_index = np.logical_not(matched_detections_index)

                remaining_detections = torch.tensor(box2[remaining_detections_index].reshape(-1, 4), device = device)

                results[index].boxes.xyxy = torch.cat((results[index].boxes.xyxy, remaining_detections))
                results[index].boxes.xyxyn = torch.cat((results[index].boxes.xyxyn, remaining_detections/np.array([width, height, width, height])))

                remaining_detections[:, 0], remaining_detections[:,1], remaining_detections[:,2], remaining_detections[:,3] = (remaining_detections[:,0] + remaining_detections[:,2])/2,  (remaining_detections[:,1] + remaining_detections[:,3])/2, remaining_detections[:,2] - remaining_detections[:,0], remaining_detections[:,3] - remaining_detections[:,1]

                results[index].boxes.xywh = torch.cat((results[index].boxes.xyxy, remaining_detections))
                results[index].boxes.xywhn = torch.cat((results[index].boxes.xyxyn, remaining_detections/np.array([width, height, width, height])))

                results[index].boxes.conf = torch.cat((results[index].boxes.conf, torch.tensor([0.8 for _ in range(len(remaining_detections))], device=device)))
                results[index].boxes.cls = torch.cat((results[index].boxes.cls, torch.tensor([len(self.CLASS_NAMES_DICT)-1 for _ in range(len(remaining_detections))], device=device)))

        return results


    def write_video(self, frame, results, persist, video, fps, save_video, save_dir):
        if persist:
            for result in results:
                frame = result.orig_img

                frame = self.Annotator([frame], [result])
                self.frame_memory.append(frame)
                if save_video:
                    if self.VideoWriter:
                        self.VideoWriter.write(frame)
                    else:
                        height, width = results[0].boxes.orig_shape
                        VIDEO_CODEC = "MJPG"
                        self.VideoWriter = cv2.VideoWriter(os.path.join(save_dir, video), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))

                        self.VideoWriter.write(frame)

        else:
            if save_video:
                height, width = results[0].boxes.orig_shape
                VIDEO_CODEC = "MJPG"
                self.VideoWriter = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))

            for result in results:
                frame = result.orig_img

                frame = self.Annotator([frame], [result])
                self.frame_memory.append(frame)
                if save_video:
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

        # box2 area
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        return inter_area / (box2_area + box1_area[:, None] - inter_area + eps)


    def register_tracker(self, source, persist, tracker):
        if not hasattr(self, 'trackers'):
            tracker = check_yaml(tracker)
            cfg = IterableSimpleNamespace(**yaml_load(tracker))
            assert cfg.tracker_type in ['bytetrack', 'botsort'], \
                f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
            trackers = []
            for _ in range(source['data'].batch_size):
                tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
                trackers.append(tracker)
            
            self.trackers = trackers
        else:
            if persist == False:
                self.reset_tracker()
                
                tracker = check_yaml(tracker)
                cfg = IterableSimpleNamespace(**yaml_load(tracker))
                assert cfg.tracker_type in ['bytetrack', 'botsort'], \
                    f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
                trackers = []
                for _ in range(source['data'].batch_size):
                    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
                    trackers.append(tracker)
                
                self.trackers = trackers
    

    def update_tracker(self, results, frames, batch_size):
        for i in range(batch_size):
            det = results[i].boxes.to(torch.device('cpu')).numpy()
            if len(det) == 0:
                continue
            tracks = self.trackers[i].update(det, frames[i], len(self.CLASS_NAMES_DICT)-1)
            if len(tracks) == 0:
                continue
            idx = tracks[:, -1].astype(int)
            results[i] = results[i][idx]
            results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))

    def save_txt(self, results, txt_file):
        for index, result in enumerate(results):
            texts = []
            boxes = result.boxes
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *d.xywhn.view(-1))
                line += (conf, ) * True + (() if id is None else (id, ))
                texts.append(('%g ' * len(line)).rstrip() % line)

            with open(os.path.join(txt_file, f"{'0'+(6-len(str(self.nb_frame+index))) + str(self.nb_frame+index)}.txt"), 'a') as f:
                f.writelines(text + '\n' for text in texts)
        
        LOGGER.info(f"Labels saved at {txt_file}")
    
    def save_frame(self, frames, frame_dir):
        for index, frame in enumerate(frames):
            cv2.imwrite(os.path.join(frame_dir, f"{'0'+(6-len(str(self.nb_frame+index))) + str(self.nb_frame+index)}.jpg"), frame)
        
        LOGGER.info(f"Frames saved at {frame_dir}")

