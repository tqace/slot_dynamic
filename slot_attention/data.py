import json
import torch
import ipdb
import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from slot_attention.utils import compact


class CLEVRERDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        clevr_transforms: Callable,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.data_path = os.path.join(data_root, split)
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        #assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(self, index: int):
        video_path = self.files[index]
        image_files = os.listdir(video_path)
        image_files.sort(key=self.sort_key)
        imgs = []
        for image_file in image_files:
            img = Image.open(os.path.join(video_path,image_file))
            img = img.convert("RGB")
            img = self.clevr_transforms(img)
            imgs.append(img.unsqueeze(0))
            if len(imgs)>40:
                break
        return torch.cat(imgs[1:],dim=0)

    def __len__(self):
        return len(self.files)

    def sort_key(self,e):
        return int(e[:-4].split('_')[-1])

    def get_files(self) -> List[str]:
        paths: List[Optional[str]] = []
        total_videos = os.listdir(self.data_path)
        for video in total_videos:
            paths.append(os.path.join(self.data_path,video))
        return sorted(compact(paths))


class CLEVRERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_batch_size: int,
        val_batch_size: int,
        clevr_transforms: Callable,
        num_workers: int,
        num_train_images: Optional[int] = None,
        num_val_images: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.clevr_transforms = clevr_transforms
        self.num_workers = num_workers
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images

        self.train_dataset = CLEVRERDataset(
            data_root=self.data_root,
            clevr_transforms=self.clevr_transforms,
            split="train",
        )
        self.val_dataset = CLEVRERDataset(
            data_root=self.data_root,
            clevr_transforms=self.clevr_transforms,
            split="val",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            #collate_fn=self.collate_fun,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            #collate_fn=self.collate_fun,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    '''
    def collate_fun(self,items):
        batch_size = len(items)
        C,H,W = items[0][0].shape
        max_len = max([len(item) for item in items])
        input_tensor = torch.zeros(batch_size,max_len,C,H,W)
        is_pad = torch.zeros(batch_size,max_len)
        for i in range(batch_size):
            is_pad[i][len(items[i]):]=1
            for j in range(len(items[i])):
                input_tensor[i][j]=items[i][j]
        return {'x':input_tensor,'is_pad':is_pad}
    '''

class CLEVRERTransforms(object):
    def __init__(self, resolution: Tuple[int, int]):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
                transforms.Resize(resolution),
            ]
        )

    def __call__(self, input, *args, **kwargs):
        return self.transforms(input)
