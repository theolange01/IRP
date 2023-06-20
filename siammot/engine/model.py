import os

import torch
from torch import nn
from siammot.utils import LOGGER, RANK
from siammot.train import train
from siammot.utils.torch_utils import select_device

from siammot.configs.default import cfg
from siammot.model.rcnn import build_siammot
from siammot.engine.trainer import make_optimizer, make_lr_scheduler, do_train
from siammot.data.dataloader import build_train_data_loader

from siammot.model.rcnn import build_siammot
from siammot.utils.torch_utils import smart_inference_mode, model_info

class SiamMOT:
    """
    todo
    """


    def __init__(self, config_file, model=''):

        self.cfg = cfg
        self.cfg.merge_from_file(config_file)
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics

        if model:
            self._load(model)
        else:
            self._new()
    
    def __call__(self, source=None, **kwargs):
        """Calls the 'track' function with given arguments to perform object detection."""
        return self.track(source, **kwargs)
    
    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _new(self):
        self.cfg.MODEL.WEIGHT = ""
        self.model = build_siammot(self.cfg)

    def _load(self, cfg, model):
        self.cfg.MODEL.WEIGHT = model
        self.model = build_siammot(cfg)
        #todo find a way to check the backbone used in model_weight

    def reset_siammot_tracking_memory(self):
        self.model.reset_siammot_status()

    @smart_inference_mode()
    def reset_weights(self):
        """
        Resets the model modules parameters to randomly initialized values, losing all training information.
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self
    
    def info(self, detailed=False, verbose=True):
        """
        Logs model info.
        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(model=self.model, detailed=detailed, verbose=verbose)
    
    @smart_inference_mode()
    def track(self, source=None, stream=False, **kwargs):
        """
        todo
        """
        if source is None:
            raise FileNotFoundError("No source given")
        
        # todo check if video or folder of images
        # else error

        

        pass

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        todo
        """
        pass


    def train(self, data = "", epochs=100, batch_size=16, device=None, ckpt="", train_dir = 'runs/train'):
        """
        todo
        """

        if not data:
            raise AttributeError("Dataset required but missing, i.e pass 'data=(Train_DIR, Val_DIR)' or 'data=(Train_DIR, ) ")
        
        if isinstance(data, str):
            data = (data, )

        i = 1
        while os.exists(train_dir):
            train_dir = train_dir + "i"
            i += 1

        cfg.SOLVER.EPOCHS = epochs
        cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = batch_size

        if device in ['gpu']:
            device = 'cuda:0'

        self.device = select_device(device, batch_size)

        self.model.to(device)

        optimizer = make_optimizer(cfg, self.model)
        scheduler = make_lr_scheduler(cfg, optimizer)

        LOGGER.info("Creating the dataloader")
        data_loaders = build_train_data_loader(cfg, data, batch_size)
        if data_loaders['val']:
            LOGGER.info("Trained and Validation DataLoaders created")
        else:
            LOGGER.info("DataLoaders created, training only")

        start_epochs = None
        if ckpt:
            try:
                checkpoint = torch.load(ckpt)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epochs = checkpoint['epochs']
                LOGGER.info("WARNING ⚠️ The model saved in the checkpoint has not the same architecture as the one created given the config_file")
                del checkpoint
            except:
                LOGGER.warning("WARNING ⚠️ The model saved in the checkpoint has not the same architecture as the one created given the config_file")

        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        epochs = cfg.SOLVER.MAX_ITER

        cfg.freeze()
        with open(os.path.join(train_dir, "SiamMOT_" + cfg.MODEL.BACKBONE.CONV_BODY.upper() + ".yaml"), 'w') as f:
            f.write(cfg.dump())

        self.model, self.train_history = do_train(self.model, data_loaders, optimizer, scheduler, device, epochs, checkpoint_period, train_dir, start_epochs)



    def to(self, device):
        """
        Sends the model to the given device.
        Args:
            device (str): device
        """
        self.model.to(device)

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None