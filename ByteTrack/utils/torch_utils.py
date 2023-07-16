# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import os
import platform
import random
from copy import deepcopy
from pathlib import Path
import pkg_resources as pkg

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from siammot.utils import emojis, LOGGER, RANK


def check_version(current: str = '0.0.0',
                  minimum: str = '0.0.0',
                  pinned: bool = False) -> bool:
    """
    Check current version against the required minimum version.
    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
    Returns:
        (bool): True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result

TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_2_0 = check_version(torch.__version__, minimum='2.0')


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


def select_device(device='', batch=0, newline=False, verbose=True):
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = f'SiamMOT Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == 'cuda':
            device = '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no ' \
                      'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                             f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                             f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                             f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                             f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                             f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                             f'{install}')

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch > 0 and batch % n != 0:  # check batch_size is divisible by device_count
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available() and TORCH_2_0:
        # Prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if verbose and RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)



def model_info(model, detailed=False, verbose=True, imgsz=[1920, 1280]):
    """Model information. imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]."""
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info('%5g %40s %9s %12g %20s %10.3g %10.3g %10s' %
                        (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype))
    flops = get_flops_with_torch_profiler(model, imgsz)

    fused = ' (fused)' if getattr(model, 'is_fused', lambda: False)() else ''
    fs = f', {flops:.1f} GFLOPs' if flops else ''
    yaml_file = getattr(model, 'yaml_file', '') or getattr(model, 'yaml', {}).get('yaml_file', '')
    model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or 'Model'
    LOGGER.info(f'{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}')
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops_with_torch_profiler(model, imgsz=[1920, 1280]):
    # Compute model FLOPs (thop alternative)
    model = de_parallel(model)
    p = next(model.parameters())
    stride = (max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32) * 2  # max stride
    im = torch.zeros((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
    with torch.profiler.profile(with_flops=True) as prof:
        with torch.no_grad():
            model.eval()
            model(im)
    flops = sum(x.flops for x in prof.key_averages()) / 1E9
    imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
    flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    return flops


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            LOGGER.warning('WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.')


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


class EarlyStopping:
    """
    Early stopping class that stops training when a specified number of epochs have passed without improvement.
    """

    def __init__(self, patience=50):
        """
        Initialize early stopping object

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `patience=300` or use `patience=0` to disable EarlyStopping.')
        return stop