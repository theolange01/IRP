from typing import List
import numpy as np


def bbs_iou(box_xywh_1: List, box_xywh_2: List):
    """
    Compute iou matrix between two lists of Bounding Boxes
    bbox is in the format of xywh
    """

    if len(box_xywh_1) == 0 or len(box_xywh_2) == 0:
        return np.zeros((len(box_xywh_1), len(box_xywh_2)))

    # compute the area of union regions
    area1 = box_xywh_1[:, 2] * box_xywh_1[:, 3]
    area2 = box_xywh_2[:, 2] * box_xywh_2[:, 3]

    # to xyxy
    box_xyxy_1 = np.zeros_like(box_xywh_1)
    box_xyxy_2 = np.zeros_like(box_xywh_2)
    box_xyxy_1[:, :2] = box_xywh_1[:, 0:2]
    box_xyxy_2[:, :2] = box_xywh_2[:, 0:2]
    box_xyxy_1[:, 2:] = box_xywh_1[:, :2] + box_xywh_1[:, 2:]
    box_xyxy_2[:, 2:] = box_xywh_2[:, :2] + box_xywh_2[:, 2:]

    lt = np.maximum(box_xyxy_1[:, None, :2], box_xyxy_2[:, :2])  # [N,M,2]
    rb = np.minimum(box_xyxy_1[:, None, 2:], box_xyxy_2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)

    return iou