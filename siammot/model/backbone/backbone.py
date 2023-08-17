# IRP SiamMOT Tracker

import os

import torch
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import timm


def build_backbone(cfg):
    """
    Build the backbone of the object detection part of the Tracker
    The Backbone can be any timm convolutional network backbone used for object detection.
    A FPN layer is applied on top of the backbone to get more features

    When weights are given in the configuration file, they will be loaded if possible

    Args:
        cfg (yacs.config.CfgNode): model configuration

    Results:
        backbone_with_fpn (torch.nn.Module): An object detection backbone with a FPN layer
    
    """

    # Create the backbone 
    backbone = timm.create_model(cfg.MODEL.BACKBONE.CONV_BODY, pretrained=True, features_only=True, out_indices=cfg.MODEL.BACKBONE.OUT_INDICES)

    backbone_with_fpn = BackboneWithFPN(backbone, backbone.return_layers, backbone.feature_info.channels(), cfg.MODEL.BACKBONE.OUT_CHANNEL)
    
    # Load the weights if possible
    if os.path.exists(cfg.MODEL.BACKBONE.WEIGHT):
        try:
            backbone_with_fpn.load_state_dict(torch.load(cfg.MODEL.BACKBONE.WEIGHT, map_location=torch.device('cpu')))
        except:
            try:
                backbone_with_fpn.load_state_dict(torch.load(cfg.MODEL.BACKBONE.WEIGHT, map_location=torch.device('cuda:0')))
            except:
                pass

    return backbone_with_fpn