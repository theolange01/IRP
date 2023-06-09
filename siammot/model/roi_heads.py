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
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, features, proposals, image_shapes, original_image_shapes, targets=None, track_memory=None, given_detection=None):
        losses = {}

        # Given detections are detections obtained with another detection model, I could use YOLO for detection are then track with SiamMOT
        # They need to be adapted to any size changes ...
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
            if len(given_detection[0]) > 0: 
                x, detections, loss_box = self.box(features, given_detection, image_shapes, targets)
            else:
                x = features
                detections = given_detection
                loss_box = {}
        losses.update(loss_box)


        if self.cfg.MODEL.TRACK_ON:
            tmp_proposals = []
            for proposal in proposals:
                tmp_proposals.append(BoxList(proposal, image_shapes)) # 1 BoxList per img

            proposals = tmp_proposals
            # proposals: List[BoxList]

            if targets:
                tmp_targets = []
                for target in range(targets):
                    box = BoxList(target['boxes'], image_shapes)
                    box.add_field('labels', [target['labels']])
                    tmp_targets.append(box)

                targets = tmp_targets
                # targets: List[BoxList]

            y, tracks, loss_track = self.track(features, proposals, targets, track_memory)
            losses.update(loss_track)

            # solver is only needed during inference
            if not self.training:

                # detections to List[BoxList]
                tmp_detections = []
                for detection in detections:
                    tmp_detections.append({'boxes': detection.bboxes,
                                           'labels': detection.get_field('labels'),
                                           'scores': detection.get_field('scores')})
                
                detection = tmp_detections

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

        if self.cfg.MODEL.TRACK_HEAD.TRACKTOR:
            scores = det_scores
        else:
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
    if not cfg.MODEL.RPN_ONLY:
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