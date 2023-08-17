# IRP SiamMOT Tracker
# Copied from # Ultralytics YOLO 🚀, AGPL-3.0 license: https://github.com/ultralytics/ultralytics

import logging.config
import os
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # siammot
DEFAULT_CFG_PATH = ROOT / 'configs/default.py'
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
LOGGING_NAME = 'siammot'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ('Darwin', 'Linux', 'Windows'))  # environment booleans

# Settings
torch.set_printoptions(linewidth=320, precision=4, profile='default')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)

def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.
    This filter is particularly useful for ensuring compatibility with Windows terminals
    that may not support the display of emojis in log messages.
    """

    def filter(self, record):
        """Filter logs by emoji unicode characters on windows."""
        record.msg = emojis(record.msg)
        return super().filter(record)


# Set logger
set_logging(LOGGING_NAME, verbose=True)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if WINDOWS:  # emoji-safe logging
    LOGGER.addFilter(EmojiFilter())

def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']