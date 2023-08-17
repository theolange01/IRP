# IRP SiamMOT Tracker

import os

import torch
from torchvision.models.detection.rpn import RegionProposalNetwork, AnchorGenerator, RPNHead 


def build_rpn(cfg, in_channels):
    """
    Function to build the RPN layer. 
    Given a set of features, the RPN layer will propose Region of Interest for the Box predictor
    
    Args:
        cfg (yacs.config.CfgNode): Model configuration
        in_channels (int): Size of the input channel

    Returns:
        rpn (torch.nn.Module): RPN module
    """
    
    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = cfg.MODEL.RPN.ASPECT_RATIOS * len(anchor_sizes)    
    
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios) 
    
    rpn_head = RPNHead(in_channels, rpn_anchor_generator.num_anchors_per_location()[0], 2)
    
    rpn_pre_nms_top_n = dict(training=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST)
    rpn_post_nms_top_n = dict(training=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST)

    # Create the RPN module
    rpn = RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.RPN.POSITIVE_FRACTION,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        cfg.MODEL.RPN.NMS_THRESH,
        score_thresh=cfg.MODEL.RPN.SCORE_THRESH,
    )

    # Load weights when possible
    if os.path.exists(cfg.MODEL.RPN.WEIGHT):
        try:
            rpn.load_state_dict(torch.load(cfg.MODEL.RPN.WEIGHT, map_location=torch.device('cpu')))
        except:
            try:
                rpn.load_state_dict(torch.load(cfg.MODEL.RPN.WEIGHT, map_location=torch.device('cuda:0')))
            except:
                pass
    
    return rpn