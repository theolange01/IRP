# IRP SiamMOT Tracker
# Implements the Generalized R-CNN for SiamMOT
# Adapted from https://github.com/amazon-science/siam-mot

from typing import List, Tuple
import os

import torch
from torch import nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

from siammot.model.backbone.backbone import build_backbone
from siammot.model.rpn.rpn import build_rpn
from siammot.model.roi_heads import build_roi_heads

from siammot.model.box_head.box_head import FeatureExtractor

from siammot.utils import LOGGER


class SiamMOT(nn.Module):
    """
    Main class for R-CNN. Currently supports boxes and tracks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and
             computes detections / tracks from it.

    Args:
        cfg (yacs.config.CfgNode): Default Model Configuration
    
    Attributes:
        backbone (nn.Module): SiamMOT backbone Module
        rpn (nn.Module): SiamMOT RPN module
        roi_heads (nn.Module): SiamMOT Roi_Heads Module
        track_memory (List): Tracking memory
        transform_train (): Transformation to apply during tracking
        transform_test (): Transformation to apply during testing
        use_faster_rcnn (bool): Whether not use the PyTorch ResNet50 Faster R-CNN

    Methods:
        __init__(): Initialise the SiamMOT model
        forward(): Perform a forward pass 
        flush_memory(): Reset Tracking memory
        reset_siammot_status(): Reset SiamMOT memory
    """

    def __init__(self, cfg):
        """
        Initialise the SiamMOT model.

        Args:
            cfg (yacs.config.CfgNode): Default Model Configuration
        """
        
        super(SiamMOT, self).__init__()

        self.transform_train = GeneralizedRCNNTransform(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        
        self.transform_test = GeneralizedRCNNTransform(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST,
                                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

        self.use_faster_rcnn = cfg.MODEL.USE_FASTER_RCNN

        # Whether to use the PyTorch ResNet50 Faster R-CNN
        if self.use_faster_rcnn:
            model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
            self.backbone = model.backbone
            self.rpn = model.rpn
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

            # replace the classifier with a new one, that has
            # num_classes which is user-defined
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            self.roi_heads.box = FeatureExtractor(model.roi_heads, ['box_head'])

            del model

            # Load the model's weights
            if os.path.exists(cfg.MODEL.BACKBONE.WEIGHT):
                try:
                    self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.WEIGHT, map_location=torch.device('cpu')))
                except:
                    try:
                        self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.WEIGHT, map_location=torch.device('cuda:0')))
                    except:
                        pass
            
            if os.path.exists(cfg.MODEL.RPN.WEIGHT):
                try:
                    self.rpn.load_state_dict(torch.load(cfg.MODEL.RPN.WEIGHT, map_location=torch.device('cpu')))
                except:
                    try:
                        self.rpn.load_state_dict(torch.load(cfg.MODEL.RPN.WEIGHT, map_location=torch.device('cuda:0')))
                    except:
                        pass
        
            if os.path.exists(cfg.MODEL.ROI_BOX_HEAD.WEIGHT):
                try:
                    self.roi_heads.box.load_state_dict(torch.load(cfg.MODEL.ROI_BOX_HEAD.WEIGHT, map_location=torch.device('cpu')))
                except:
                    try:
                        self.roi_heads.box.load_state_dict(torch.load(cfg.MODEL.ROI_BOX_HEAD.WEIGHT, map_location=torch.device('cuda:0')))
                    except:
                        pass

        else:
            # Use the Faster R-CNN given in cfg.
            self.backbone = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.track_memory = None

    def flush_memory(self, cache=None):
        """Reset Track memory."""
        self.track_memory = cache

    def reset_siammot_status(self):
        """Reset SiamMOT Status."""
        self.flush_memory()
        self.roi_heads.reset_roi_status()

    def forward(self, images, targets=None, given_detection=None):
        """
        Perform a forward pass.

        Args:
            images (List[torch.Tensor]): a batch of pair of two images, each images in PIL format Tensor[3, height, width]
            targets list[Dict[str, Tensor]]: contains:
                - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
                ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
                - labels (``Int64Tensor[N]``): the class label for each ground-truth box
            given_detection (list of dictionary): contains:
                - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
                ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
                - labels (``Int64Tensor[N]``): the class label for each ground-truth box
                - confidence score
        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # Apply the transformation to the input images
        if self.training:
            images, targets = self.transform_train(images, targets)
        else:
            images, targets = self.transform_test(images, targets)

        # Obtain features map
        features = self.backbone(images.tensors)

        # Apply the RPN on the features map
        if self.use_faster_rcnn:
            proposals, proposal_losses = self.rpn(images, features, targets)
        else:
            proposals, proposal_losses, objectness = self.rpn(images, features, targets)
        # proposals : List[Tensor[N,4]], proposal_losses: Dict[str, float]

        # Apply the roi_heads
        x, results, roi_losses = self.roi_heads(features,
                                                proposals,
                                                images.image_sizes,
                                                targets,
                                                self.track_memory,
                                                given_detection)
        
        # Update the tracking memory
        if not self.training:
            self.flush_memory(cache=x)

        # Change the format of the results
        tmp_results = []
        for result in results:
            tmp_results.append({'boxes': result.bbox, 
                                'labels': result.get_field('labels'),
                                'scores': result.get_field('scores'),
                                'ids': result.get_field('ids')})
        
        detections = tmp_results

        # Postprocess detections
        if self.training:
            result = self.transform_train.postprocess(detections, images.image_sizes, original_image_sizes)
        else:
            result = self.transform_test.postprocess(detections, images.image_sizes, original_image_sizes)

        if self.training: 
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return result, losses

        return result


def build_siammot(cfg):
    """Build the SiamMOT model."""

    siammot = SiamMOT(cfg)
    LOGGER.info(f"{cfg.MODEL.BACKBONE.CONV_BODY.upper()} backbone")

    if cfg.MODEL.WEIGHT:
        try:
            siammot.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location=torch.device('cpu')))
            LOGGER.info("Model Weights loaded")
        except:
            try:
                siammot.load_state_dict(torch.load(cfg.MODEL.WEIGHT, map_location=torch.device('cuda:0')))
                LOGGER.info("Model Weights loaded")
            except:
                cfg.MODEL.WEIGHT = ""
                LOGGER.warning("WARNING ⚠️ The model's architecture and the weights are not compatible\n Using new model instead")


    return siammot