from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F

from dataset import SegmentationDataset, UnlabeledDataset






def get_multiclass_dataloader(data_dir, 
                            image_folder = 'Labeled/Images',
                            mask_folder = 'Labeled/Split_Mask/all',
                            unlabeled_folder = 'Unlabeled',
                            fraction = 0.2,
                            batch_size = 4,
                            num_classes=None,
                            seed=None,
                            distributed=False):

    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        'Train': SegmentationDataset(data_dir,
                            image_folder=image_folder,
                            mask_folder=mask_folder,
                            seed=seed,
                            fraction=fraction,
                            subset='Train',
                            transforms=data_transforms,
                            num_classes=num_classes),
        'Unlabeled': UnlabeledDataset(data_dir,
                            image_folder=unlabeled_folder,
                            transforms=data_transforms,
                            num_classes=num_classes),
        'Validation': SegmentationDataset(data_dir,
                            image_folder=image_folder,
                            mask_folder=mask_folder,
                            seed=seed,
                            fraction=fraction,
                            subset='Validation',
                            transforms=data_transforms,
                            num_classes=num_classes),
        'Test': SegmentationDataset(data_dir,
                            image_folder=image_folder,
                            mask_folder=mask_folder,
                            seed=seed,
                            fraction=fraction,
                            subset='Test',
                            transforms=data_transforms,
                            num_classes=num_classes)
    }

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(image_datasets['Train'], shuffle=True)
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(image_datasets['Unlabeled'], shuffle=True)
    else:
        train_sampler = None
        unlabeled_sampler = None

    dataloaders = {
        'Train': DataLoader(image_datasets['Train'],
                    batch_size=batch_size,
                    sampler=train_sampler,
                    num_workers=4,
                    pin_memory=True),
        'Unlabeled': DataLoader(image_datasets['Unlabeled'],
                    batch_size=batch_size,
                    sampler=unlabeled_sampler,
                    num_workers=4,
                    pin_memory=True),
        'Validation': DataLoader(image_datasets['Validation'],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True),
        'Test': DataLoader(image_datasets['Test'],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True)
    }

    return dataloaders, image_datasets['Test'].image_names