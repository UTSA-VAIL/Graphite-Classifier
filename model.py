from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch
from torchsummary import summary
import segmentation_models_pytorch as smp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def deeplabv3(outputchannels=1):

    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    return model

def vgg11(outputchannels=1):
    model = smp.Unet('vgg11', encoder_weights='imagenet', classes=outputchannels)
    return model

def vgg19(outputchannels=1):
    model = smp.Unet('vgg19', encoder_weights='imagenet', classes=outputchannels)
    return model

def vgg19_bn(outputchannels=1):
    model = smp.Unet('vgg19_bn', encoder_weights='imagenet', classes=outputchannels)
    return model

def resnet18(outputchannels=1):
    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=outputchannels)
    return model

def resnet152(outputchannels=1):
    model = smp.Unet('resnet152', encoder_weights='imagenet', classes=outputchannels)
    return model

def resnet101(outputchannels=1):
    model = smp.Unet('resnet101', encoder_weights='imagenet', classes=outputchannels)
    return model

def resnet50(outputchannels=1):
    model = smp.Unet('resnet50', encoder_weights='imagenet', classes=outputchannels)
    return model

def effnet_b0(outputchannels=1):
    model = smp.Unet('efficientnet-b0', encoder_weights='imagenet', classes=outputchannels)
    return model

def effnet_b3(outputchannels=1):
    model = smp.Unet('efficientnet-b3', encoder_weights='imagenet', classes=outputchannels)
    return model

def dpn68(outputchannels=1):
    model = smp.Unet('dpn68', encoder_weights='imagenet', classes=outputchannels)
    return model

def dpn98(outputchannels=1):
    model = smp.Unet('dpn98', encoder_weights='imagenet', classes=outputchannels)
    return model

def dpn107(outputchannels=1):
    model = smp.Unet('dpn107', encoder_weights='imagenet+5k', classes=outputchannels)
    return model

def dpn92(outputchannels=1):
    model = smp.Unet('dpn92', encoder_weights='imagenet+5k', classes=outputchannels)
    return model

def dpn131(outputchannels=1):
    model = smp.Unet('dpn131', encoder_weights='imagenet', classes=outputchannels)
    return model

def senet154(outputchannels=1):
    model = smp.Unet('senet154', encoder_weights='imagenet', classes=outputchannels)
    return model

def densenet121(outputchannels=1):
    model = smp.Unet('densenet121', encoder_weights='imagenet', classes=outputchannels)
    return model


def get_model(model_type, outputchannels=1):
    models = {'vgg11'    : vgg11,
              'vgg19'    : vgg19,
              'vgg19-bn' : vgg19_bn,
              'resnet18' : resnet18,
              'resnet50' : resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152,
              'deeplabv3': deeplabv3,
              'effnet-b0': effnet_b0, 
              'effnet-b3': effnet_b3,
              'dpn68'    : dpn68,
              'dpn98'   : dpn98,
              'dpn92'   : dpn92,
              'dpn107'   : dpn107,
              'dpn131'   : dpn131,
              'senet154' : senet154,
              'densenet121': densenet121
            }
    if model_type in models:
        model = models[model_type](outputchannels)
        print(f'Using model: {model_type}')
    else:
        model = models['resnet18'](outputchannels)
        print(f"'{model_type}' is not a valid model. Using default: 'resnet18'")
    return model
