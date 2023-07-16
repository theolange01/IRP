# Use that https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py
import os
from copy import deepcopy

from .clear import CLEAR
from .hota import HOTA
from .count import Count

import numpy as np
import torch

def prepare_data(gts, results):
    data = {}
    """
    Data = dict
        - num_timesteps (int): number of frames
        - num_tracker_dets (int): 
        - num_gt_dets (int):
        - num_tracker_ids (int): Number of different ID predicted by the tracker
        - num_gt_ids (int): Total Number of different ID present in the video
        - gt_ids (): List of 1D array with ID of objects appearing in frame t
        - tracker_ids (): List of 1D array with ID of objects tracked in frame t
        - similarity_scores (List[np.array]): list of size num_timesteps containing 2D arrays with the IoU similarity between each gt and detections for each frame
    
    """
    num_timesteps = len(results)
    data['num_timesteps'] = num_timesteps

    num_tracking_dets = 0
    num_gt_dets = 0

    _gt_ids = []
    _tracker_ids = []

    gt_ids = []
    tracker_ids = []

    similarity = []

    for i in range(num_timesteps):

        result = results[i]
        gt = gts[i]

        tracker_id = result.boxes.id.cpu().numpy().astype(int)
        tracker_ids.append(tracker_id)
        num_tracking_dets += len(tracker_id)

        gt_id = gt['ids']
        gt_ids.append(gt_id)
        num_gt_dets += len(gt_id)

        for id in tracker_id:
            if id not in _tracker_ids:
                _tracker_ids.append(id)
        
        for id in gt_id:
            if id not in _gt_ids:
                _gt_ids.append(id)
        
        similarity.append(_calculate_similarities(gt['boxes'].to(torch.device('cpu')).numpy(), result.boxes.xyxy.to(torch.device('cpu').numpy())))
        # compute IoU

    data['num_tracker_dets'] = num_tracking_dets
    data['num_gt_dets'] = num_gt_dets
    data['num_tracker_ids'] = len(_tracker_ids)
    data['num_gt_ids'] = len(_gt_ids)

    data['tracker_ids'] = tracker_ids
    data['gt_ids'] = gt_ids

    data['similarity_scores'] = similarity

    return data

def _calculate_similarities(gt_dets_t, tracker_dets_t):
        similarity_scores = _calculate_box_ious(gt_dets_t, tracker_dets_t)
        return similarity_scores


def _calculate_box_ious(bboxes1, bboxes2, box_format='xyxy', eps=1e-7):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """
        if box_format in 'xywh':
            # layout: (x0, y0, w, h)
            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in 'xyxy':
            raise (Exception('box_format %s is not implemented' % box_format))

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2.T

        # Intersection area
        inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                    (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

        # box2 area
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        return inter_area / (box2_area + box1_area[:, None] - inter_area + eps)


def evaluate(gt, results):
    data = prepare_data(gt, results)

    count_metrics = Count().eval_sequence(data)
    clear_metrics = CLEAR().eval_sequence(data)
    HOTA_metrics = HOTA().eval_sequence(data)

    return {'Count': count_metrics, 'Clear': clear_metrics, 'HOTA': HOTA_metrics}