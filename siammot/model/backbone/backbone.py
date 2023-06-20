import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models.detection.backbone_utils import BackboneWithFPN
import timm

def build_backbone(cfg):

    if not os.path.exists(cfg.MODEL.WEIGHT):
        backbone = timm.create_model(cfg.MODEL.BACKBONE.CONV_BODY, pretrained=True, features_only=True, out_indices=cfg.MODEL.BACKBONE.OUT_INDICES)
    
    else:
        backbone = timm.create_model(cfg.MODEL.BACKBONE.CONV_BODY, pretrained=False, features_only=True, out_indices=cfg.MODEL.BACKBONE.OUT_INDICES)
    
    backbone_with_fpn = BackboneWithFPN(backbone, backbone.return_layers, backbone.feature_info.channels(), cfg.MODEL.BACKBONE.OUT_CHANNEL)
    
    return backbone_with_fpn