from yacs.config import CfgNode as CN

cfg = CN()

#----------------MODEL----------------#
cfg.MODEL = CN()
cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN'
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.RPN_ONLY = False

cfg.MODEL.WEIGHT = ""

#----------------BACKBONE----------------#
cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.CONV_BODY = "dla60"
cfg.MODEL.BACKBONE.WEIGHT = ""
cfg.MODEL.BACKBONE.OUT_INDICES = (2,3,4,5)
cfg.MODEL.BACKBONE.OUT_CHANNEL = 256

cfg.MODEL.BACKBONE.WEIGHT = ""


#----------------INPUT----------------#
cfg.INPUT = CN()

# Size of the smallest side of the image during training
cfg.INPUT.MIN_SIZE_TRAIN = 1280
# Maximum size of the side of the image during training
cfg.INPUT.MAX_SIZE_TRAIN = 1920
# Size of the smallest side of the image during testing
cfg.INPUT.MIN_SIZE_TEST = 1280
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = 1920

cfg.INPUT.MOTION_LIMIT = 0.1
cfg.INPUT.COMPRESSION_LIMIT = 50
cfg.INPUT.MOTION_BLUR_PROB = 0.5
cfg.INPUT.AMODAL = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
cfg.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
cfg.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
cfg.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
cfg.MODEL.GROUP_NORM.EPSILON = 1e-5


#----------------RPN----------------#
cfg.MODEL.RPN = CN()
cfg.MODEL.RPN.USE_FPN = True
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
cfg.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
cfg.MODEL.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)
# RPN anchor aspect ratios
cfg.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
cfg.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
cfg.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
cfg.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST = 1000
# Number of top scoring RPN proposals to keep after applying NMS
cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
cfg.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
cfg.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 300
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 300
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
cfg.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
cfg.MODEL.RPN.SCORE_THRESH = 0.0
# Custom rpn head, empty to use default conv or separable conv
cfg.MODEL.RPN.RPN_HEAD = ""

cfg.MODEL.RPN.WEIGHT = ""


#----------------ROI Heads----------------#
cfg.MODEL.ROI_HEADS = CN()
cfg.MODEL.ROI_HEADS.USE_FPN = True
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
cfg.MODEL.ROI_HEADS.BOX_NMS_THRESH = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


cfg.MODEL.ROI_BOX_HEAD = CN()
cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "TwoMLPHead"
cfg.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81 # todo
# Hidden layer dimension when using an MLP for the RoI box head
cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024

# GN
cfg.MODEL.ROI_BOX_HEAD.USE_GN = False

# Dilation
cfg.MODEL.ROI_BOX_HEAD.DILATION = 1
cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


#----------------Tracking----------------#
cfg.MODEL.TRACK_ON = True
cfg.MODEL.TRACK_HEAD = CN()
cfg.MODEL.TRACK_HEAD.TRACKTOR = False
cfg.MODEL.TRACK_HEAD.POOLER_SCALES = (0.25, 0.125, 0.0625, 0.03125)
cfg.MODEL.TRACK_HEAD.POOLER_RESOLUTION = 15
cfg.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO = 2

cfg.MODEL.TRACK_HEAD.PAD_PIXELS = 512
# the times of width/height of search region comparing to original bounding boxes
cfg.MODEL.TRACK_HEAD.SEARCH_REGION = 2.0
# the minimal width / height of the search region
cfg.MODEL.TRACK_HEAD.MINIMUM_SREACH_REGION = 0
cfg.MODEL.TRACK_HEAD.MODEL = 'EMM'

# solver params
cfg.MODEL.TRACK_HEAD.TRACK_THRESH = 0.4
cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH = 0.6
cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH = 0.4
# maximum number of frames that a track can be dormant
cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES = 20

# track proposal sampling
cfg.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE = 256
cfg.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD = 0.65
cfg.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD = 0.35

cfg.MODEL.TRACK_HEAD.IMM = CN()
# the feature dimension of search region (after fc layer)
# in comparison to that of target region (after fc layer)
cfg.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM_MULTIPLIER = 2
cfg.MODEL.TRACK_HEAD.IMM.FC_HEAD_DIM = 256

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