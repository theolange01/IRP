import argparse
import os
from typing import Tuple

import torch

from siammot.utils import LOGGER
from siammot.utils.torch_utils import select_device

from siammot.configs.default import cfg
from siammot.model.rcnn import build_siammot
from siammot.engine.trainer import make_optimizer, make_lr_scheduler, do_train
from siammot.data.dataloader import build_train_data_loader


def train(cfg, model_weight="", source="", device=None, epochs=100, batch_size=16, train_dir="runs/train", ckpt=None):
    if os.path.exists(model_weight):
        cfg.MODEL.WEIGHT = model_weight

    i = 1
    while os.exists(train_dir):
        train_dir = train_dir + "i"
        i += 1

    cfg.SOLVER.EPOCHS = epochs
    cfg.SOLVER.VIDEO_CLIPS_PER_BATCH = batch_size 

    # build model
    model = build_siammot(cfg, device)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    LOGGER.info("Creating the dataloader")
    data_loaders = build_train_data_loader(cfg, source, batch_size)
    if data_loaders['val']:
        LOGGER.info("Trained and Validation DataLoaders created")
    else:
        LOGGER.info("DataLoaders created, training only")

    start_epochs = None
    if ckpt:
        try:
            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
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

    model, history = do_train(model, data_loaders, optimizer, scheduler, device, epochs, checkpoint_period, train_dir, start_epochs)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="PyTorch SiamMOT Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--model", default="", metavar="FILE", help="path to the model weight", type=str)
    parser.add_argument("--source", type=Tuple(str), help="Path to the training data, can be (Train, ) or (Train, Val)")
    parser.add_argument("--device", default=None, help='device to run on, i.e "cpu" or "cuda:0"', type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--checkpoint", default="", help="Path to a previous checkpoint to resume training", type=str)
    parser.add_argument("--train-dir", default="runs/train", help="training folder where training artifacts are dumped", type=str)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file) 
    

    if args.device in ['gpu']:
        args.device = 'cuda:0'

    device = select_device(args.device, args.batch_size)

    _, _ = train(cfg, args.model, args.source, device, args.epochs, args.batch_size, args.train_dir, args.ckpt)

    LOGGER.info(f"Trainig successful\nResults available at '{args.train_dir}'")


if __name__ == "__main__":
    main()