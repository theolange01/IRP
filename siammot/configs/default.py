# IRP SiamMOT Tracker 
# Default model settings, training settings and hyperparameters for training

from yacs.config import CfgNode as CN
cfg = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
cfg.MODEL.USE_FASTER_RCNN = False
cfg.MODEL.WEIGHT = "" # (str, Optional) Path to the pretrained weight of a model


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------

cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.CONV_BODY = "dla34" # (str) name of the Backbone used, need to be timm model name
cfg.MODEL.BACKBONE.OUT_INDICES = (2,3,4,5) # (Tuple(int)) The index of the features to be used byt the RPN layer
cfg.MODEL.BACKBONE.OUT_CHANNEL = 128 # (int) Out Channel of the Backbone
cfg.MODEL.BACKBONE.WEIGHT = "" # (str, Optional) Path to a pretrained version of the backbone


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

cfg.INPUT = CN()
cfg.INPUT.MIN_SIZE_TRAIN = 640 # (int) Size of the smallest side of the image during training
cfg.INPUT.MAX_SIZE_TRAIN = 960 # (int) Maximum size of the side of the image during training
cfg.INPUT.MIN_SIZE_TEST = 1280 # (int) Size of the smallest side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = 1920 # (int) Maximum size of the side of the image during testing
cfg.INPUT.AMODAL = False # (bool) Whether to clip the bounding box beyond image boundary

# Data Augmentation -----------------------------------------------------------
# cfg.INPUT.MOTION_LIMIT = 0.05
# cfg.INPUT.COMPRESSION_LIMIT = 50
# cfg.INPUT.MOTION_BLUR_PROB = 1.0
# cfg.INPUT.BRIGHTNESS = 0.1
# cfg.INPUT.CONTRAST = 0.1
# cfg.INPUT.SATURATION = 0.1
# cfg.INPUT.HUE = 0.1

# Flips -----------------------------------------------------------------------
# cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
# cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0


# -----------------------------------------------------------------------------
# Group Norm options
# -----------------------------------------------------------------------------

cfg.MODEL.GROUP_NORM = CN()

cfg.MODEL.GROUP_NORM.DIM_PER_GP = -1 # (int) Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
cfg.MODEL.GROUP_NORM.NUM_GROUPS = 32 # (int) Number of groups in GroupNorm (-1 if using DIM_PER_GP)
cfg.MODEL.GROUP_NORM.EPSILON = 1e-5 # (float) GroupNorm's small constant in the denominator


# -----------------------------------------------------------------------------
# RPN
# -----------------------------------------------------------------------------

cfg.MODEL.RPN = CN()
cfg.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512) # (Tuple(int)) Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
cfg.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0) # (Tuple(float)) RPN anchor aspect ratios

# (float) Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
cfg.MODEL.RPN.FG_IOU_THRESHOLD = 0.7

#(float) Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
cfg.MODEL.RPN.BG_IOU_THRESHOLD = 0.3

cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256 # (int) Total number of RPN examples per image
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5 # (float) Target fraction of foreground (positive) examples per RPN minibatch

# (int) Number of top scoring RPN proposals to keep before applying NMS
cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000

# (int) Number of top scoring RPN proposals to keep after applying NMS
cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000

cfg.MODEL.RPN.NMS_THRESH = 0.7 # (float) NMS threshold used on RPN proposals
cfg.MODEL.RPN.MIN_SIZE = 0 # (int) Proposal height and width both need to be greater than RPN_MIN_SIZE

# (int) Number of top scoring RPN proposals to keep after combining proposals from all FPN levels
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 300
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 300

cfg.MODEL.RPN.SCORE_THRESH = 0.0 # (float) NMS threshold used for postprocessing the RPN proposals
cfg.MODEL.RPN.WEIGHT = "" # (str, Optional) Path to a pretrained version of the RPN


# -----------------------------------------------------------------------------
# RoI Heads
# -----------------------------------------------------------------------------

cfg.MODEL.ROI_HEADS = CN()
cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5 # (float) Overlap threshold for an RoI to be considered foreground
cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5 # (float) Overlap threshold for an RoI to be considered background

# Default weights on (dx, dy, dw, dh) for normalising bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25 # (float) Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)


# Test Model Only -------------------------------------------------------------

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.05

cfg.MODEL.ROI_HEADS.BOX_NMS_THRESH = 0.5 # (float) Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 200 # (int) Maximum number of detections to return per image 


cfg.MODEL.ROI_BOX_HEAD = CN()
cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "TwoMLPHead" # (str) Name of the feature extractor used. Cannot be changed
cfg.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor" # (str) Name of the predictor used. Cannot be changed
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 5 # (int) 
cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2 # (int) 
cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.25, 0.125) # (0.25, 0.125, 0.0625, 0.03125)
cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 3 # (int) Number of classes
cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 256 # (int) Hidden layer dimension when using an MLP for the RoI box head
cfg.MODEL.ROI_BOX_HEAD.WEIGHT = "" # (str, Optional) Path to a pretrained version of the RPN


# -----------------------------------------------------------------------------
# Tracking
# -----------------------------------------------------------------------------

cfg.MODEL.TRACK_ON = True # (bool) Whether to activate the tracking part of the model
cfg.MODEL.TRACK_HEAD = CN()
cfg.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125) # (0.25, 0.125, 0.0625, 0.03125) 
cfg.MODEL.TRACK_HEAD.POOLER_RESOLUTION = 7 # (int)
cfg.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO = 2 # (int)

cfg.MODEL.TRACK_HEAD.PAD_PIXELS = 256 # 512
cfg.MODEL.TRACK_HEAD.SEARCH_REGION = 2.0 # (float) the times of width/height of search region comparing to original bounding boxes
cfg.MODEL.TRACK_HEAD.MINIMUM_SEARCH_REGION = 0 # (int) the minimal width / height of the search region

cfg.MODEL.TRACK_HEAD.MODEL = 'EMM' # (str), The Model of the Track Head, only EMM implemented. 

# solver params
cfg.MODEL.TRACK_HEAD.TRACK_THRESH = 0.4 # (float)
cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH = 0.6 # (float)
cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = 0.4 # (float)
cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES = 20 # (int) Maximum number of frames that a track can be dormant

# track proposal sampling
cfg.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE = 300
cfg.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD = 0.65
cfg.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD = 0.35


cfg.MODEL.TRACK_HEAD.EMM = CN()
# Use_centerness flag only activates during inference
cfg.MODEL.TRACK_HEAD.EMM.USE_CENTERNESS = True
cfg.MODEL.TRACK_HEAD.EMM.POS_RATIO = 0.25
cfg.MODEL.TRACK_HEAD.EMM.HN_RATIO = 0.25
cfg.MODEL.TRACK_HEAD.EMM.TRACK_LOSS_WEIGHT = 1.
# The ratio of center region to be positive positions
cfg.MODEL.TRACK_HEAD.EMM.CLS_POS_REGION = 0.8
# The lower this weight, it allows large motion offset during inference
# Setting this param to be small (e.g. 0.1) for datasets that have fast motion,
# such as caltech roadside pedestrian
cfg.MODEL.TRACK_HEAD.EMM.COSINE_WINDOW_WEIGHT = 0.4


# -----------------------------------------------------------------------------
# Video
# -----------------------------------------------------------------------------

# all video-related parameters
cfg.VIDEO = CN()
# the length of video clip for training/testing
cfg.VIDEO.TEMPORAL_WINDOW = 1000
# the temporal sampling frequency for training
cfg.VIDEO.TEMPORAL_SAMPLING = 100
cfg.VIDEO.RANDOM_FRAMES_PER_CLIP = 2


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

cfg.INFERENCE = CN()
cfg.INFERENCE.USE_GIVEN_DETECTIONS = False
# The length of clip per forward pass
cfg.INFERENCE.CLIP_LEN = 1


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------

cfg.SOLVER = CN()
cfg.SOLVER.EPOCHS = 100
cfg.SOLVER.OPTIMIZER = 'SGD'

cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.BIAS_LR_FACTOR = 2

cfg.SOLVER.MOMENTUM = 0.9

cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.WEIGHT_DECAY_BIAS = 0

cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (30000,40000)

cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.WARMUP_METHOD = "linear"

cfg.SOLVER.CHECKPOINT_PERIOD = 5
cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = 16
cfg.SOLVER.TEST_PERIOD = 0

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
cfg.SOLVER.IMS_PER_BATCH = 16

