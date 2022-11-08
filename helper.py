import numpy as np
import torch
from sklearn.utils import class_weight
import copy
from itertools import tee
import torch.nn.functional as F

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



def get_multi_class_weights(train_dataloader, num_classes):
    #get class weights
    total_counts = np.zeros(num_classes)
    for batch in train_dataloader:
        sample = batch['mask']
        class_sample_count = np.unique(sample, return_counts=True)[1]
        total_counts += class_sample_count
    print(total_counts / total_counts.sum())
    class_weights = torch.FloatTensor(total_counts.sum() / (total_counts * num_classes)).cuda()
    class_weights = F.normalize(class_weights, dim=0)
    print(class_weights)
    return class_weights



