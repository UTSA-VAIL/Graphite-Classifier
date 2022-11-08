from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch
import torch.nn.functional as F


class SegmentationDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 num_classes: int = None
                 ) -> None:

        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            
            image_num = int(np.ceil(len(self.image_list) * (1 - self.fraction)))
            mask_num = int(np.ceil(len(self.mask_list) * (1 - self.fraction)))
            test_image_num = int(np.ceil(len(self.image_list) * (1 - self.fraction / 2)))
            test_mask_num = int(np.ceil(len(self.mask_list) * (1 - self.fraction / 2)))
            if subset == "Train":
                self.image_names = self.image_list[:image_num]
                self.mask_names = self.mask_list[:mask_num]
            elif subset == 'Test':
                self.image_names = self.image_list[image_num:test_image_num]
                self.mask_names = self.mask_list[mask_num:test_mask_num]
            elif subset == 'Validation':
                self.image_names = self.image_list[test_image_num:]
                self.mask_names = self.mask_list[test_mask_num:]
    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,"rb") as mask_file:
            image = Image.open(image_file)
            image = image.convert("RGB")

            mask = Image.open(mask_file)
            mask = mask.convert("L")

            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                mask = np.array(sample['mask'])
                sample['mask'] = torch.from_numpy(mask).long()
            return sample


class UnlabeledDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 image_folder: str,
                 transforms: Optional[Callable] = None,
                 num_classes: int = None
                 ) -> None:

        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder

        self.image_names = np.array(sorted(image_folder_path.glob("*")))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        with open(image_path, "rb") as image_file:
            image = Image.open(image_file)
            image = image.convert("RGB")
            
            sample = {"image": image}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
            return sample