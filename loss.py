from webbrowser import GenericBrowser
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchgeometry.losses import DiceLoss, FocalLoss
import sklearn

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class DiceCCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCCELoss, self).__init__()

    def forward(self, inputs, targets, weights):
        #dice = DiceLoss()
        #print(inputs.shape, targets.shape)
        dice = GeneralizedDiceLoss()
        cce = nn.CrossEntropyLoss(weight=weights)
        dice_loss = dice(inputs, targets)
        cce_loss = cce(inputs, targets)
        loss = 1 + ((CE_RATIO * cce_loss) + ((1 - CE_RATIO) * dice_loss))
        return loss





class GeneralizedDiceLoss(nn.Module):

    def __init__(self, epsilon=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        inputs = torch.argmax(inputs, dim=1)

        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)

        weights = targets.sum(-1)
        weights = 1 / (weights**2).clamp(min=self.epsilon)
        weights.requires_grad = False

        intersect = (inputs * targets).sum(-1)
        intersect = intersect * weights

        denominator = (inputs + targets).sum(-1)
        denominator = (denominator * weights).clamp(min=self.epsilon)

        return 1 - torch.mean(2 * (intersect.sum() / denominator.sum()))


    