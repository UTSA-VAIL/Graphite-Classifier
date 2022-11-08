
import copy
import csv
import os
import time

import numpy as np
from simplejson import OrderedDict
import torch
from tqdm import tqdm
import pathlib
import shutil
import torch.nn.functional as F
import sklearn



def alpha_weight(num_epochs, epoch):
    if epoch < int(0.10 * num_epochs): 
        return 0
    elif epoch >= int(0.90 * num_epochs): 
        return 1
    else: 
        return epoch / (num_epochs - int(0.10 * num_epochs))





def train_multiclass_model(model, criterion, dataloaders, optimizer, scheduler, metrics, exp_dir, num_epochs, local_rank, class_weights, distributed):
    print(exp_dir)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    if local_rank == 0:
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.mkdir(exp_dir)

        fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
            [f'Train_{m}' for m in metrics.keys()] + \
            [f'Validation_{m}' for m in metrics.keys()]
        with open(os.path.join(exp_dir, f'log.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        if local_rank == 0:
            ep_msg = f'Epoch {epoch}/{num_epochs}'
            print(ep_msg)
            print('-' * 10)

            batchsummary = {a: [] for a in fieldnames}
        losses = {}
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
                if distributed:
                    dataloaders['Train'].sampler.set_epoch(epoch)
            else:
                model.eval()
                if local_rank != 0:
                    continue

            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].cuda()
                masks = sample['mask'].cuda()

                # set grad if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    if isinstance(outputs, OrderedDict):
                        outputs = outputs['out']
                    y_pred = outputs
                    y_true = masks
                    loss = criterion(y_pred, y_true, class_weights)
                    for name, metric in metrics.items():
                        results = metric(y_pred, y_true)
                        if local_rank == 0:
                            batchsummary[f'{phase}_{name}'].append(results)


                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            if local_rank == 0:
                losses[f'{phase}_Loss'] = loss
                batchsummary['epoch'] = epoch
                batchsummary[f'{phase}_loss'] = round(loss.item(), 4)
            if phase == 'Validation':
                scheduler.step()
        if local_rank == 0:
            for field in fieldnames[1:]:
                if type(batchsummary[field]) is list:
                    batchsummary[field] = round(torch.mean(torch.stack(batchsummary[field]), dim=0).item(), 4)
            print(batchsummary)
            with open(os.path.join(exp_dir, f'log.csv'), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(batchsummary)
                if phase == 'Validation' and loss < best_loss:
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
    if local_rank == 0:
        print('Lowest Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
    return model







def train_semi_multiclass_model_2(model, criterion, dataloaders, optimizer, scheduler, metrics, bpath, num_epochs, device, class_weights, distributed):
    exp_dir = f"./{pathlib.Path(bpath).stem}/"
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.mkdir(exp_dir)
    summary_writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard') )
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    fieldnames = ['epoch', 'Train_loss', 'Validation_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Validation_{m}' for m in metrics.keys()]
    with open(os.path.join(exp_dir, f'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    begin_ul_train = False

    for epoch in range(1, num_epochs + 1):
        ep_msg = f'Epoch {epoch}/{num_epochs}'
        print(ep_msg)
        print('-' * 10)

        batchsummary = {a: [] for a in fieldnames}
        losses = {}
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()
            ul_dataloader = iter(dataloaders['Unlabeled'])
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].cuda()
                masks = sample['mask'].cuda()
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    y_pred = outputs
                    y_true = masks
                    l_loss = criterion(y_pred, y_true, class_weights)

                    if phase == 'Train':
                        optimizer.zero_grad()
                        l_loss.backward()
                        optimizer.step()


                    ul_loss = 0
                    if begin_ul_train:
                        if phase == 'Train':
                            steps = int(len(ul_dataloader) / len(dataloaders['Train']))
                        else:
                            steps = 1
                        for step in range(0, steps):
                            ul_batch = next(ul_dataloader)
                            ul_inputs = ul_batch['image'].cuda()
                            model.eval()
                            ul_y_true = torch.argmax(model(ul_inputs), dim=1)
                            model.train()
                            ul_y_pred = model(ul_inputs)
                            ul_loss = alpha_weight(num_epochs, epoch) * criterion(ul_y_pred, ul_y_true, class_weights)
                            if phase == 'Train':
                                optimizer.zero_grad()
                                ul_loss.backward()
                                optimizer.step()


                    loss = l_loss + ul_loss

                    for name, metric in metrics.items():
                        results = metric(y_pred, y_true)
                        batchsummary[f'{phase}_{name}'].append(results)

            losses[f'{phase}_Loss'] = loss
            batchsummary['epoch'] = epoch
            batchsummary[f'{phase}_loss'] = round(loss.item(), 4)
            print(f'{phase} Loss: {loss:.4f}, {phase} L_Loss: {l_loss:.4f}, {phase} UL_Loss: {ul_loss:.4f}')
            if phase == 'Validation':
                scheduler.step()
        summary_writer.add_scalars(f'Loss', losses, epoch)
        for field in fieldnames[1:]:
            if type(batchsummary[field]) is list:
                batchsummary[field] = round(torch.mean(torch.stack(batchsummary[field]), dim=0).item(), 4)
        if batchsummary['Validation_miou'] > 0.85:
            begin_ul_train = True
        print(batchsummary)
        with open(os.path.join(exp_dir, f'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            if phase == 'Validation':# and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
    summary_writer.close()
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model