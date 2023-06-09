import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models.detection.backbone_utils import BackboneWithFPN
import timm

def build_backbone(cfg):
    backbone = timm.create_model(cfg.MODEL.BACKBONE.CONV_BODY, pretrained=True, features_only=True, out_indices=cfg.MODEL.BACKBONE.OUT_INDICES)
    
    if os.path.exists(cfg.MODEL.BACKBONE.WEIGHT):
        backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.WEIGHT, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    
    backbone_with_fpn = BackboneWithFPN(backbone, backbone.return_layers, backbone.feature_info.channels(), cfg.MODEL.BACKBONE.OUT_CHANNEL)
    
    return backbone_with_fpn