from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead

def build_rpn(cfg, in_channels):
    anchor_sizes = cfg.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = cfg.MODEL.RPN.ASPECT_RATIOS * len(anchor_sizes)    
    
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios) 
    
    rpn_head = RPNHead(in_channels, rpn_anchor_generator.num_anchors_per_location()[0], 2)
    
    rpn_pre_nms_top_n = dict(training=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST)
    rpn_post_nms_top_n = dict(training=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN, testing=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST)

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
    
    return rpn