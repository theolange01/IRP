# IRP SiamMOT Tracker
# Adapted from https://github.com/amazon-science/siam-mot

import torch
from torch import nn

from siammot.model.box_head.box_head import build_roi_box_head
from siammot.model.track_head.track_head import build_track_head
from siammot.model.track_head.track_utils import build_track_utils
from siammot.model.track_head.track_solver import builder_tracker_solver

from siammot.structures.bounding_box import BoxList
from siammot.structures.boxlist_ops import cat_boxlist


class CombinedROIHeads(nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single head.

    Args:
        cfg (yacs.config.CfgNode): Default Model Configuration
        heads (List): List of heads to combine
    
    Attributes:
        cfg (yacs.config.CfgNode): Default Model Configuration
        box (nn.Module): Box classification Heads
        track (nn.Module): Tracking Head
        solver (nn.Module): Tracking Solver Head
    
    Methods:
        __init__(): Initialise CombinedROIHeads
        forward(): forward pass
        reset_roi_status(): Reset tracking memory
        _refine_tracks(): Refine tracking results
    """

    def __init__(self, cfg, heads):
        """
        Initialise the CombinedROIHeads module.

        Args:
            cfg (yacs.config.CfgNode): Default Model Configuration
            heads (List): List of heads to combine
        """
        
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, features, proposals, image_shapes, targets=None, track_memory=None, given_detection=None):
        """
        Forward Function.

        Args:
            features (Dict): FPN features map
            proposals (List[List[int, 4]]): List of proposed ROI for each images
            image_shapes (List[Tuple[int, int]]): List of frame shape
            targets (Dict[str, torch.Tensor]): Dictionary giving the ground truth results
            track_memory (): Tracking memory from the previous frame
            given_detection (Any): Detections from another object detector
        """
        
        
        losses = {}

        if given_detection is None:
            x, detections, loss_box = self.box(features, proposals, image_shapes, targets)
            # x: Feature map, (Tensor[max_detection, MLP Head DIM])
            # detections: (list of dictionary), containing:
            #   - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
            #   - labels (Int64Tensor[N]): the class label for each box,
            #   - scores (FloatTensor[N]): the confidence score for each box
            # loss_box: (Dict[str, float])

        else:
            # adjust provided detection
            # It needs to be in the correct format
            if len(given_detection[0]) > 0: 
                x, detections, loss_box = self.box(features, given_detection, image_shapes, targets)
            else:
                x = features
                detections = given_detection
                loss_box = {}
        losses.update(loss_box)

        # detections to List[BoxList]
        tmp_detections = []
        for index, detection in enumerate(detections):
            box = BoxList(detection['boxes'], image_shapes[index])
            box.add_field('labels', detection['labels'])
            box.add_field('scores', detection['scores'])
            try:
                box.add_field('ids', detection['ids'])
            except:
                box.add_field('ids', torch.Tensor([-1 for _ in range(len(detection['boxes']))], device = detection['labels'].device))
            tmp_detections.append(box)
        
        detections = tmp_detections

        del tmp_targets

        # If the tracking is activated
        if self.cfg.MODEL.TRACK_ON:
            features = list(features.values())

            # proposals to List[BoxList]
            tmp_proposals = []
            for index, proposal in enumerate(proposals):
                tmp_proposals.append(BoxList(proposal, image_shapes[index])) # 1 BoxList per img

            proposals = tmp_proposals
            
            del tmp_proposals

            if targets:
                # targets to List[BoxList]
                tmp_targets = []
                for index, target in enumerate(targets):
                    box = BoxList(target['boxes'], image_shapes[index])
                    box.add_field('labels', target['labels'])
                    try:
                        box.add_field('ids', target['ids'])
                    except:
                        box.add_field('ids', torch.Tensor([i for i in range(len(target['labels']))]).to(target['labels'].device))
                    tmp_targets.append(box)

                targets = tmp_targets
                
                del tmp_targets

            # Perform tracking
            y, tracks, loss_track = self.track(features, proposals, targets, track_memory)
            losses.update(loss_track)

            # solver is only needed during inference
            if not self.training:

                if tracks is not None:
                    tracks = self._refine_tracks(features, tracks)
                    detections = [cat_boxlist(detections + tracks)]

                detections = self.solver(detections)

                # get the current state for tracking
                x = self.track.get_track_memory(features, detections)


        return x, detections, losses


    def reset_roi_status(self):
        """
        Reset the status of ROI Heads
        """
        if self.cfg.MODEL.TRACK_ON:
            self.track.reset_track_pool()


    def _refine_tracks(self, features, tracks):
        """
        Use box head to refine the bounding box location
        The final vis score is an average between appearance and matching score
        """
        if len(tracks[0]) == 0:
            return tracks[0]
        track_scores = tracks[0].get_field('scores') + 1.
        # track_boxes = tracks[0].bbox
        _, tracks, _ = self.box(features, tracks)
        det_scores = tracks[0].get_field('scores')
        det_boxes = tracks[0].bbox

        scores = (det_scores + track_scores) / 2.
        boxes = det_boxes

        r_tracks = BoxList(boxes, image_size=tracks[0].size, mode=tracks[0].mode)
        r_tracks.add_field('scores', scores)
        r_tracks.add_field('ids', tracks[0].get_field('ids'))
        r_tracks.add_field('labels', tracks[0].get_field('labels'))

        return [r_tracks]


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    roi_heads = []
    roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))  

    if cfg.MODEL.TRACK_ON:
        track_utils, track_pool = build_track_utils(cfg)
        roi_heads.append(("track", build_track_head(cfg, track_utils, track_pool)))
        # solver is a non-learnable layer that would only be used during inference
        roi_heads.append(("solver", builder_tracker_solver(cfg, track_pool)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads