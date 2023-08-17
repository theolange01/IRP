# IRP SiamMOT Tracker

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch

from siammot.utils import colorstr, LOGGER
from siammot.utils.torch_utils import select_device

from siammot.configs.default import cfg
from siammot.model.rcnn import build_siammot
from siammot.engine.trainer import make_optimizer, make_lr_scheduler, do_train
from siammot.data.dataloader import build_train_data_loader


def train(cfg, model_weight="", source="", device=None, epochs=100, batch_size=16, train_dir="runs/train", ckpt=None):
    """
    Train a SiamMOT model.

    Args:
        cfg (yacs.config.CfgNode): Default model config
        model_weight (str): Path to the model's weights
        source (str): Path to the dataset
        device (torch.device): Device to use for training
        epochs (int): Number of epochs
        batch_size (int): Number of data per batch
        train_dir (str): Path to save the training results
        ckpt (str): Path to a checkpoint save of a previous training step
    
    Returns:
        model (Any): Trained SiamMOT model
        history (Dict[str, List]): Training Loss evolution
    """
    
    # Update the configuration
    if os.path.exists(model_weight):
        cfg.MODEL.WEIGHT = model_weight

    # Whether to use the Pytorch ResNet50 Faster R-CNN detection model
    if cfg.MODEL.USE_FASTER_RCNN:
        cfg.MODEL.BACKBONE.CONV_BODY = "Resnet50"
        cfg.MODEL.BACKBONE.OUT_CHANNEL = 256

    i = 1
    _train_dir = train_dir
    while os.path.exists(_train_dir):
        _train_dir = train_dir + str(i)
        i += 1

    train_dir = _train_dir
    del _train_dir
        
    os.makedirs(train_dir, exist_ok=True)

    cfg.SOLVER.EPOCHS = epochs
    cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = batch_size 

    # build SiamMOT model
    model = build_siammot(cfg)
    model.to(device)

    # Define optimiser and lr scheduler for training
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    LOGGER.info("Creating the dataloaders...")
    data_loaders = build_train_data_loader(cfg, (source, ), batch_size)
    if data_loaders['val']:
        LOGGER.info(f"{colorstr('DataLoaders:')} Training and Validation ✅")
    else: 
        LOGGER.info(f"{colorstr('DataLoaders:')} Training only ✅")

    start_epochs = 0
    if ckpt: # Load checkpoint 
        try:
            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epochs = checkpoint['epochs']
            LOGGER.info("The Checkpoint data has been successfully loaded ✅")
            del checkpoint
        except:
            LOGGER.warning("WARNING ⚠️ The model saved in the checkpoint has not the same architecture as the one created given the config_file")

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # Save model configuration
    cfg.freeze()
    if not cfg.MODEL.USE_FASTER_RCNN:
        with open(os.path.join(train_dir, "SiamMOT_" + cfg.MODEL.BACKBONE.CONV_BODY.upper() + ".yaml"), 'w') as f:
            f.write(cfg.dump())
    else:
        with open(os.path.join(train_dir, "SiamMOT_" + "fasterrcnn_resnet50_fpn".upper() + ".yaml"), 'w') as f:
            f.write(cfg.dump())

    # Train model
    model, history = do_train(model, data_loaders, optimizer, scheduler, device, epochs, checkpoint_period, train_dir, start_epochs)

    LOGGER.info(f"Trainig successful\nResults available at '{train_dir}'")

    return model, history


def main():
    """
    Use: python3 train.py --config-file 'siammot/configs/default.yaml' --source 'IRP_dataset/train'
    
    """
    parser = argparse.ArgumentParser(description="PyTorch SiamMOT Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--model", default="", metavar="FILE", help="path to the model weight", type=str)
    parser.add_argument("--source", type=str, help="Path to the training data, can be (Train, ) or (Train, Val)")
    parser.add_argument("--device", default=None, help='device to run on, i.e "cpu" or "cuda:0"', type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--checkpoint", default="", help="Path to a previous checkpoint to resume training", type=str)
    parser.add_argument("--train-dir", default="runs/train", help="training folder where training artifacts are dumped", type=str)

    args = parser.parse_args()

    try: 
        cfg.merge_from_file(args.config_file) 
    except:
        pass
    
    if args.device in ['gpu']:
        args.device = 'cuda:0'

    device = select_device(args.device, args.batch_size)

    _, _ = train(cfg, args.model, args.source, device, args.epochs, args.batch_size, args.train_dir, args.checkpoint)


if __name__ == "__main__":
    main()