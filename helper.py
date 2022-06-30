import numpy as np
import torch
from sklearn.utils import class_weight
import copy
from itertools import tee

def per_class_mIoU(targets, predictions, info=False): 
    unique_labels = np.unique(targets)
    ious = list()
    for index in unique_labels:
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection)/np.sum(union)
        ious.append(iou_score)
    if info:
        print ("per-class mIOU: ", ious)
    return np.average(ious)


def get_binary_class_weights(train_dataloader):
    num_pos = 0
    num_neg = 0
    for batch in train_dataloader:
        pos = torch.sum(batch['mask'])
        num_neg += torch.sum(torch.ones_like(batch['mask'])) - pos
        num_pos += pos

    c_weights = num_neg / num_pos
        
    return c_weights


def get_multi_class_weights(train_dataloader, num_classes):
    print(num_classes)
    weights = []
    for c in range(0, num_classes):
        num_pos = 0
        num_neg = 0
        for batch in train_dataloader:
            print(c, torch.unique(batch['mask'], return_counts=True))
            pos = torch.sum(batch['mask'])
            num_neg += torch.sum(torch.ones_like(batch['mask'])) - pos
            num_pos += pos

        #weights.append(num_neg / num_pos)
    
    return (weights)


def one_hot(targets, num_classes):  
    # targets_extend=targets.clone()
    # #targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    # one_hot = torch.FloatTensor(targets_extend.size(0), num_classes, targets_extend.size(2), targets_extend.size(3)).zero_()
    # one_hot.scatter_(1, targets_extend, 1) 
    # return one_hot

    _, height, width = targets.shape
    one_hot = torch.zeros(num_classes, height, width)
    #print(one_hot.shape, targets.shape)
    return one_hot.scatter_(1, targets, 1.0)


def one_hot_numpy(targets, num_classes):
    b = np.zeros((num_classes, targets.size[0], targets.size[1]))
    b[np.arange(targets.size), targets] = 1
    one_hot_tensor = torch.from_numpy(b) 
    one_hot_tensor.requires_grad=True
    return one_hot_tensor


