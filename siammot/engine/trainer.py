import os
import time
import copy

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

from siammot.utils import LOGGER, TQDM_BAR_FORMAT, colorstr
from siammot.configs.default import cfg

def make_optimizer(cfg, model):
    """
    Create and define the optimizer for the training
    
    Args:
        cfg (yacs.config.CfgNode): Configuration for the model training
        model (torch.nn.Module): The model to be trained

    Returns:
        optimizer (torch.optim.optimizer): The optimizer to use during training 
    """
    
    params = []
    g = [], []

    # Get the list of parameters and check their key
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            g[0].append(value)
        else:
            g[1].append(value)
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # Define the optimizer
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={cfg.SOLVER.MOMENTUM}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay})')
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    """
    Create the LR Scheduler
    
    Args:
        cfg (yacs.config.CfgNode): Configuration for training
        optimizer (torch.optim.optimizer): The optimizer to be used during training

    Returns:
        scheduler (torch.optim.lr_scheduler): The Learning Rate to be applied during training
    """
    
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA
    )



def do_train(model, dataloaders, optimizer, scheduler, device, num_epochs, checkpoint_period, train_dir, starting_epoch=0):
    """
    
    Args:
        model (torch.nn.Module): The model to train
        dataloaders (torch.utils.data.DataLoader): The Videos clips dataloaders
        optimizer (torch.optim.optimizer): The optimizer to be used during training
        scheduler (torch.optim.lr_scheduler): The Learning Rate to be applied during training
        device (torch.device): The device used for the training of the model
        num_epochs (int): The number of training epochs
        checkpoint_period (int): The number of epochs between saving a checkpoint of the model state
        train_dir (str, Path): The directory to save the training data and outputs.
        starting_epoch (int): Only used when resuming training. Starting epoch to resume training.
    
    Returns:
        model (torch.nn.Module): The trained model
        train_history (Dict(str, List[float])): The losses history
    """
    LOGGER.info("")
    LOGGER.info(f"{colorstr('Model Training:')} Starting training for {num_epochs} epochs...")
    
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    losses = {}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    if device == torch.device('cuda:0'):
        scaler = GradScaler()

    if starting_epoch:
        LOGGER.info(f"{colorstr('Resume Training:')} Training restarting from epoch {starting_epoch}")
    
    LOGGER.info("")

    for epoch in range(starting_epoch, num_epochs):
        LOGGER.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        LOGGER.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if dataloaders[phase]:
                    model.eval()   # Set model to evaluate mode
                else:
                    continue

            running_loss = 0.0

            # Iterate over data.
            for video, targets, _ in tqdm(dataloaders[phase]):
                inputs = [v.to(device) for v in video]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # zero the parameter gradients
                optimizer.zero_grad()

                if device == torch.device('cuda:0'):
                    with autocast():
                        _, loss_dict = model(inputs, targets)
                        loss = sum(value if value==value else 0 for value in loss_dict.values()) # todo, check loss size and type to get it right

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                else:
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        _, loss_dict = model(inputs, targets)
                        loss = sum(value for value in loss_dict.values()) # todo, check loss size and type to get it right

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                # statistics
                running_loss += loss.item() * len(inputs)

                if not running_loss == running_loss:
                    print(loss_dict)
                
                if len(losses.keys()):
                    for k, _ in loss_dict.items():
                        if k in losses.keys():
                            losses[k] += loss_dict[k] * len(inputs)
                        else:
                            losses[k] = loss_dict[k] * len(inputs)
                else:
                    losses.update(loss_dict)
                    for k, _ in losses.items():
                        losses[k] *= len(inputs)

                del inputs
                del targets
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            LOGGER.info('{} Loss: {:.4f} '.format(phase, epoch_loss))
            LOGGER.info("")

            # deep copy the model
            if dataloaders['val']:
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(train_dir, 'best.pt'))
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(train_dir, 'best.pt'))

            if phase == 'val':
                val_loss_history.append(epoch_loss)
            else:
                train_loss_history.append(epoch_loss)
        
        if (epoch + 1) % checkpoint_period == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(train_dir, f"checkpoint_{epoch+1}.pt"))


    time_elapsed = time.time() - since
    LOGGER.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    LOGGER.info('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'train_loss': train_loss_history, 'val_loss': val_loss_history}