# IRP SiamMOT Tracker

import os
from pathlib import Path
from typing import Union

import torch
from torch import nn

from siammot.utils import LOGGER, colorstr
from siammot.utils.torch_utils import select_device

from siammot.configs.default import cfg
from siammot.model.rcnn import build_siammot
from siammot.engine.trainer import make_optimizer, make_lr_scheduler, do_train
from siammot.data.dataloader import build_train_data_loader

from siammot.model.rcnn import build_siammot
from siammot.utils.torch_utils import smart_inference_mode


class SiamMOT:
    """
    SiamMOT object tracking model.

    Args:
        config_file (str, Path): Path to the model cfg file. Default to None
        task (str, Path): Path to the model's weight. Default to None. If not compatible with the cfg file, it won't be loaded

    Attributes:
        model (Any): The SiamMOT model object.
        cfg (yacs.config.CfgNode): The model configuration.
        device (torch.device): The device on which the model is running
        train_history (List[float]): The history of the loss during training

    Methods:
        __init__(): Initialise the SiamMOT model
        __call__(): Default call function for the tracking
        _new(): Create a new SiamMOT model from scratch
        _load(): Load a SiamMOT model
        reset_siammot_tracking_memory(): Reset the memory of the SiamMOT model
        track(): Function to perform tracking. Not Implemented 
        val(): Function to perform validation. Not Implemented
        train(): Function to train the SiamMOT model
        to(): Sends the model to the given device        
    """


    def __init__(self, config_file: Union[str, Path] ='', model: Union[str, Path] =''):
        """
        
        Args:
            config_file (str): Path to a custom config file
            model (Any): The SiamMOT model object.
        """
        
        # Load and update the config file
        self.cfg = cfg
        if config_file:
            self.cfg.merge_from_file(config_file)
        
        # Use the ResNet50 Faster R-CNN from PyTorch 
        if self.cfg.MODEL.USE_FASTER_RCNN:
            self.cfg.MODEL.BACKBONE.CONV_BODY = "Resnet50"
            self.cfg.MODEL.OUT_CHANNELS = 256    
        
        self.model = None  # model object

        # Load or create the model
        if model or self.cfg.MODEL.WEIGHT:
            self._load(model)
        else:
            self._new()
    

    def __call__(self, source=None, **kwargs):
        """Calls the 'track' function with given arguments to perform object detection."""
        return self.track(source, **kwargs)


    def _new(self):
        # Create a new Model
        self.cfg.MODEL.WEIGHT = ""
        self.model = build_siammot(self.cfg)


    def _load(self, model: Union[str, Path]):
        # Create a new model and try to load the given weights
        if model:
            self.cfg.MODEL.WEIGHT = model
        self.model = build_siammot(self.cfg)


    def reset_siammot_tracking_memory(self):
        # Reset the memory of the tracking heads
        self.model.reset_siammot_status()

    
    @smart_inference_mode()
    def track(self, source=None, stream=False, **kwargs):
        if source is None:
            raise FileNotFoundError("No source given")
        
        raise NotImplementedError()


    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        raise NotImplementedError()


    def train(self, data = "", epochs=100, batch_size=2, device=None, ckpt="", train_dir = 'runs/train'):
        """
        Train the SiamMOT Model.

        Args:
            data (str, Path): Path to the data for training
            epochs (int): Number of epochs for the training
            batch_size (int): DataLoader batch_size
            device (device): Device to be used during training, can be None, 'cpu', 'cuda:0', '0' or '0,1,2,3'
            ckpt (str, Path): Path to a checkpoint of the model to resume training. Optional.
            train_dir (str, Path): Path to save the training data
    
        Returns:
            model (Any): Trained SiamMOT model
            train_history (Dict[str, List]): Training Loss evolution
        """

        if not data:
            raise AttributeError("Dataset required but missing, i.e pass 'data=(Train_DIR, Val_DIR)' or 'data=(Train_DIR, ) ")
        
        if isinstance(data, str):
            data = (data, )

        # Check if the directory exists to create a new one if needed
        i = 1
        _train_dir = train_dir
        while os.path.exists(_train_dir):
            _train_dir = train_dir + str(i)
            i += 1

        train_dir = _train_dir
        del _train_dir
        
        os.makedirs(train_dir, exist_ok=True)

        # Select the device for training
        if device in ['gpu']:
            device = 'cuda:0'

        self.device = select_device(device=device, batch=batch_size)
        self.model.to(self.device)

        # Create the optimizer and learning rate scheduler for training
        optimizer = make_optimizer(cfg, self.model)
        scheduler = make_lr_scheduler(cfg, optimizer)

        LOGGER.info("")
        LOGGER.info("Creating the dataloaders...")
        data_loaders = build_train_data_loader(cfg, data, batch_size)
        if data_loaders['val']:
            LOGGER.info(f"{colorstr('DataLoaders:')} Training and Validation ✅")
        else: 
            LOGGER.info(f"{colorstr('DataLoaders:')} Training only ✅")

        start_epochs = 0
        # Load the chekpoint if possible to resume training
        if ckpt:
            try:
                checkpoint = torch.load(ckpt)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epochs = checkpoint['epochs']
                LOGGER.info("")
                LOGGER.info("The Checkpoint data has been successfully loaded ✅")
                del checkpoint
            except:
                LOGGER.warning("WARNING ⚠️ The model saved in the checkpoint has not the same architecture as the one created given the config_file")

        # Set the checkpoint period to frequently save the state of the model 
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

        # cfg.freeze()
        # Save the specific configuration of the model for future use
        with open(os.path.join(train_dir, "SiamMOT_" + cfg.MODEL.BACKBONE.CONV_BODY.upper() + ".yaml"), 'w') as f:
            f.write(cfg.dump())

        # Train the model
        self.model, self.train_history = do_train(self.model, data_loaders, optimizer, scheduler, self.device, epochs, checkpoint_period, train_dir, start_epochs)

    def to(self, device):
        """
        Sends the model to the given device.
        Args:
            device (str): device
        """
        self.model.to(device)