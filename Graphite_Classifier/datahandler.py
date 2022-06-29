from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F

from segdataset import SegmentationDataset




def get_binary_dataloader(data_dir, 
                            num_classes,
                            image_folder = 'Images',
                            mask_folder = 'Masks',
                            fraction = 0.2,
                            batch_size = 4,
                            seed=100):
    data_transforms = transforms.Compose([
        transforms.ToTensor()
                            ])


    dataloaders_dict = {}
    for c in range(1, num_classes):
        #if c == 0:
        #    continue
        image_datasets = {
            'Train': SegmentationDataset(data_dir,
                                image_folder=image_folder,
                                mask_folder=mask_folder + f'/{c}',
                                seed=seed,
                                fraction=fraction,
                                subset='Train',
                                transforms=data_transforms),
            'Validation': SegmentationDataset(data_dir,
                                image_folder=image_folder,
                                mask_folder=mask_folder + f'/{c}',
                                seed=seed,
                                fraction=fraction,
                                subset='Validation',
                                transforms=data_transforms),
            'Test': SegmentationDataset(data_dir,
                                image_folder=image_folder,
                                mask_folder=mask_folder + f'/{c}',
                                seed=seed,
                                fraction=fraction,
                                subset='Test',
                                transforms=data_transforms)
        }
        dataloaders = {
            'Train': DataLoader(image_datasets['Train'],
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8),
            'Validation': DataLoader(image_datasets['Validation'],
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8),
            'Test': DataLoader(image_datasets['Test'],
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8)
        }
        dataloaders_dict[c] = dataloaders
    return dataloaders_dict, image_datasets['Test'].image_names




def get_multiclass_dataloader(data_dir, 
                            image_folder = 'Labeled/Images',
                            mask_folder = 'Labeled/Masks/all',
                            fraction = 0.2,
                            batch_size = 4,
                            seed=100,
                            num_classes=None):
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
    dataloaders = {
        'Train': DataLoader(image_datasets['Train'],
                    batch_size=batch_size,
                    shuffle=True,),
        'Validation': DataLoader(image_datasets['Validation'],
                    batch_size=batch_size,
                    shuffle=True,),
        'Test': DataLoader(image_datasets['Test'],
                    batch_size=batch_size,
                    shuffle=True,)
    }

    return dataloaders, image_datasets['Test'].image_names