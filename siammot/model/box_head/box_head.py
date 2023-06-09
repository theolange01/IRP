from typing import Dict, Iterable, Callable

import torch
from torch import nn

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads

class FeatureExtractor(nn.Module):
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
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["1", "2", "3", "4", "5"], output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO)

    resolution = box_roi_pool.output_size[0]
    box_head = TwoMLPHead(in_channels * resolution**2, cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM)

    box_predictor = FastRCNNPredictor(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

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
    
    roi_box_heads = FeatureExtractor(roi_heads, ['box_head'])
    
    return roi_box_heads