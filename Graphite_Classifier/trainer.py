
import copy
import csv
import os
import time

import numpy as np
from simplejson import OrderedDict
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pathlib
import shutil
import torch.nn.functional as F
from helper import one_hot

threshold = 0.0


def train_model(model, criterion, dataloaders, optimizer, scheduler, metrics, bpath, num_epochs, device, class_num=None, num_classes=None):
    class_path = f"./{pathlib.Path(bpath).stem}/{class_num}"
    if os.path.exists(class_path):
        shutil.rmtree(class_path)
    os.mkdir(class_path)
    summary_writer = SummaryWriter(log_dir=os.path.join(class_path, 'tensorboard') )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Validation_{m}' for m in metrics.keys()]
    with open(os.path.join(class_path, f'log_{class_num}.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        ep_msg = f'Epoch {epoch}/{num_epochs}'
        if class_num:
            ep_msg += f' - Model: {class_num} of {num_classes-1}'
        print(ep_msg)
        print('-' * 10)

        # Initialize batch summary
        batchsummary = {a: [] for a in fieldnames}
        losses = {}
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                print(inputs)
                print(masks)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    if isinstance(outputs, OrderedDict):
                        outputs = outputs['out']
                    loss = criterion(outputs, masks)
                    y_pred = outputs.data.cpu()
                    y_true = masks.data.cpu().type(torch.int32)
                    for name, metric in metrics.items():
                        results = metric(y_pred, y_true)
                        batchsummary[f'{phase}_{name}'].append(results)


                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            losses[f'{phase}_Loss'] = loss
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = round(epoch_loss.item(), 4)
            print(f'{phase} Loss: {loss:.4f}')
            if phase == 'Validation':
                scheduler.step(metrics=epoch_loss)
        summary_writer.add_scalars(f'Loss_{class_num}', losses, epoch)
        for field in fieldnames[1:]:
            batchsummary[field] = round(np.mean(batchsummary[field]), 4)
        print(batchsummary)
        with open(os.path.join(class_path, f'log_{class_num}.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            if phase == 'Validation' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
    summary_writer.close()
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model





def train_multiclass_model(model, criterion, dataloaders, optimizer, scheduler, metrics, bpath, num_epochs, device, num_classes=None):
    class_path = f"./{pathlib.Path(bpath).stem}/"
    if os.path.exists(class_path):
        shutil.rmtree(class_path)
    os.mkdir(class_path)
    summary_writer = SummaryWriter(log_dir=os.path.join(class_path, 'tensorboard') )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Validation_{m}' for m in metrics.keys()]
    with open(os.path.join(class_path, f'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        ep_msg = f'Epoch {epoch}/{num_epochs}'
        print(ep_msg)
        print('-' * 10)

        # Initialize batch summary
        batchsummary = {a: [] for a in fieldnames}
        losses = {}
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    #print("Output Shape:", outputs.shape)
                    #print("Ground Truth Shape: ", masks.shape)
                    if isinstance(outputs, OrderedDict):
                        outputs = outputs['out']
                    y_pred = outputs
                    y_true = masks
                    if type(criterion) is list:
                        dice_loss = criterion[0](y_pred, y_true) 
                        CCE_loss = criterion[1](y_pred, y_true)
                        loss = dice_loss + CCE_loss
                    else:
                        loss = criterion(y_pred, y_true)
                    for name, metric in metrics.items():
                        results = metric(y_pred, y_true)
                        batchsummary[f'{phase}_{name}'].append(results)


                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            losses[f'{phase}_Loss'] = loss
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = round(epoch_loss.item(), 4)
            print(f'{phase} Loss: {loss:.4f}')
            if phase == 'Validation':
                scheduler.step(metrics=epoch_loss)
        summary_writer.add_scalars(f'Loss', losses, epoch)
        for field in fieldnames[1:]:
            if type(batchsummary[field]) is list:
                batchsummary[field] = round(torch.mean(torch.stack(batchsummary[field]), dim=0).item(), 4)
        print(batchsummary)
        with open(os.path.join(class_path, f'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            if phase == 'Validation' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
    summary_writer.close()
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model