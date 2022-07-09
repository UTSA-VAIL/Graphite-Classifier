""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch
from torchsummary import summary
import segmentation_models_pytorch as smp



def deeplabv3(outputchannels=1):

    model = models.segmentation.deeplabv3_resnet101(pretrained=False,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    return model

def resnet18(outputchannels=1):
    model = smp.Unet()
    model = smp.Unet('resnet18', encoder_weights=None, classes=outputchannels)
    return model

def resnet152(outputchannels=1):
    model = smp.Unet()
    model = smp.Unet('resnet152', encoder_weights='imagenet', classes=outputchannels)
    return model

def resnet101(outputchannels=1):
    model = smp.Unet()
    model = smp.Unet('resnet101', encoder_weights='imagenet', classes=outputchannels)
    return model

def resnet50(outputchannels=1):
    model = smp.Unet()
    model = smp.Unet('resnet50', encoder_weights='imagenet', classes=outputchannels)
    return model

def effNet(outputchannels=1):
    model = smp.Unet()
    model = smp.Unet('efficientnet-b0', encoder_weights='imagenet', classes=outputchannels)
    return model


def get_model(model_type, outputchannels=1):
    models = {'resnet18' : resnet18,
              'resnet50' : resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152,
              'deeplabv3': deeplabv3,
              'effNet'   : effNet, 
            }
    if model_type in models:
        model = models[model_type](outputchannels)
        print(f'Using model: {model_type}')
    else:
        model = models['resnet152'](outputchannels)
        print(f"'{model_type}' is not a valid model. Using default: 'resnet152'")
    return model
