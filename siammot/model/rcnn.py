"""
Implements the Generalized R-CNN for SiamMOT
"""
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
    """

    def __init__(self, cfg):
        super(SiamMOT, self).__init__()

        self.transform_train = GeneralizedRCNNTransform(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        
        self.transform_test = GeneralizedRCNNTransform(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST,
                                                  image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

        self.use_faster_rcnn = cfg.MODEL.USE_FASTER_RCNN

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
            self.backbone = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.track_memory = None

    def flush_memory(self, cache=None):
        self.track_memory = cache

    def reset_siammot_status(self):
        self.flush_memory()
        self.roi_heads.reset_roi_status()

    def forward(self, images, targets=None, given_detection=None):
        """
        images: a batch of pair of two images, each images in PIL format Tensor[3, height, width]
        targets: list[Dict[str, Tensor]], containing:
            - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        given_detection: (list of dictionary), containing:
            - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
            ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (``Int64Tensor[N]``): the class label for each ground-truth box
            - confidence score
        """
        # todo: transform given_detection into List[BoxList] if that's not the case and not None.

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

        if self.training:
            images, targets = self.transform_train(images, targets)
        else:
            images, targets = self.transform_test(images, targets)

        features = self.backbone(images.tensors)

        if self.use_faster_rcnn:
            proposals, proposal_losses = self.rpn(images, features, targets)
        else:
            proposals, proposal_losses, objectness = self.rpn(images, features, targets)
        # proposals : List[Tensor[N,4]], proposal_losses: Dict[str, float]

        # If given detection, update boxes position given the size of reshaped images
        x, results, roi_losses = self.roi_heads(features,
                                                proposals,
                                                images.image_sizes,
                                                targets,
                                                self.track_memory,
                                                given_detection)
        
        if not self.training:
            self.flush_memory(cache=x)

        tmp_results = []
        for result in results:
            tmp_results.append({'boxes': result.bbox, 
                                'labels': result.get_field('labels'),
                                'scores': result.get_field('scores'),
                                'ids': result.get_field('ids')})
        
        detections = tmp_results

        if self.training:
            result = self.transform_train.postprocess(detections, images.image_sizes, original_image_sizes)
        else:
            result = self.transform_test.postprocess(detections, images.image_sizes, original_image_sizes)

        if self.training or targets: # Get loss even during validation
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return result, losses

        return result


def build_siammot(cfg):
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