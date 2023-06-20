import os
import time
import copy

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim

from siammot.utils import LOGGER, TQDM_BAR_FORMAT, colorstr
from siammot.configs.default import cfg

def make_optimizer(cfg, model, LOGGER):
    params = []
    g = [], []
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

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={cfg.SOLVER.MOMENTUM}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={weight_decay})')
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA
    )



def do_train(model, dataloaders, optimizer, scheduler, device, num_epochs, checkpoint_period, train_dir, starting_epoch=0):
    LOGGER.info(f"Starting training for {num_epochs} epochs...")
    
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    losses = {}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    if starting_epoch != 0:
        LOGGER.info(f"Resuming training...\n Training restarting from epoch {starting_epoch}")

    for epoch in range(starting_epoch, num_epochs):
        LOGGER.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        LOGGER.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                if phase in dataloaders.keys():
                    model.eval()   # Set model to evaluate mode
                else:
                    continue

            running_loss = 0.0

            # Iterate over data.
            for video, targets, _ in tqdm(dataloaders[phase]):
                inputs = video.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

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
                running_loss += loss.item() * inputs.size(0)
                if len(losses.keys()):
                    for k, _ in losses.items():
                        losses[k] += loss_dict[k] * inputs.size(0)
                else:
                    losses.update(loss_dict)
                    for k, _ in losses.items():
                        losses[k] += loss_dict[k] * inputs.size(0)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            LOGGER.info('{} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if 'val' in dataloaders.keys():
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
        
        if epoch % checkpoint_period == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(train_dir, f"checkpoint_{epoch}.pt"))

        print()

    time_elapsed = time.time() - since
    LOGGER.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    LOGGER.info('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'train_loss': train_loss_history, 'val_loss': val_loss_history}