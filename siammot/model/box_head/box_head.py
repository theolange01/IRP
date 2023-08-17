# IRP SiamMOT Tracker

import os
from typing import Iterable, Callable

import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import *


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor module. 
    This class is used to extract the features created by the RoI box Heads. These features are used for the tracking

    Args:
        model (torch.nn.Module): The RoI_Heads model
        layers (List[str]): The list of layers from which extract the features

    Attributes:
        model (torch.nn.Module): The RoI_Heads model
        layers (List[str]): The list of layers from which extract the features
        _features (Dict[str, torch.Tensor]): A dictionnary containing the features after the forward call

    Methods:
        save_outputs_hook(layer_id: str)
        forward(features, proposals, image_shapes, targets)
    
    """
    
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, features, proposals, image_shapes, targets=None,):
        detections, detector_losses = self.model(features, proposals, image_shapes, targets)
        return self._features[self.layers[0]], detections, detector_losses



def build_roi_box_head(cfg, in_channels):
    """
    Create the RoI Box heads for the object detection part.
    The RoI Box_Heads will detect and identify the object on the images. The features used for the detection will also be returned

    Args:
        cfg (yacs.config.CfgNode): Model configuration 
        in_channels (int): The size of the input channels

    Returns:
        roi_box_heads (torch.nn.Module): The Box predictor
    """
    
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["1", "2", "3", "4", "5"], output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO)

    resolution = box_roi_pool.output_size[0]
    box_head = TwoMLPHead(in_channels * resolution**2, cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM)

    box_predictor = FastRCNNPredictor(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

    # Create the RoIHeads
    roi_heads = RoIHeads(
        box_roi_pool,
        box_head,
        box_predictor,
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
        cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS,
        cfg.MODEL.ROI_HEADS.SCORE_THRESH,
        cfg.MODEL.ROI_HEADS.BOX_NMS_THRESH,
        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG,
    )
    
    # Apply the FeatureExtractor 
    roi_box_heads = FeatureExtractor(roi_heads, ['box_head'])

    # Load the weights when possible
    if os.path.exists(cfg.MODEL.ROI_BOX_HEAD.WEIGHT):
        try:
            roi_box_heads.load_state_dict(torch.load(cfg.MODEL.ROI_BOX_HEAD.WEIGHT, map_location=torch.device('cpu')))
        except:
            try:
                roi_box_heads.load_state_dict(torch.load(cfg.MODEL.ROI_BOX_HEAD.WEIGHT, map_location=torch.device('cuda:0')))
            except:
                pass
    
    return roi_box_heads