from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import random
import pandas as pd
import os
from os import PathLike
from loguru import logger
from typing import Any
from config import PROCESSED_DATA_DIR, CSV_DATA_DIR
from typing import Literal


def data_augmentation(img: torch.Tensor) -> torch.Tensor:
    step_rotation = v2.Lambda(lambda img: F.rotate(img, angle=random.choice([0, 90, 180, 270])))
    transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomZoomOut(fill=205, side_range=[1.0, 1.3], p=0.5),
            step_rotation,
            v2.Resize((500, 500), antialias=False),

        ]
    )
    return transforms(img)


class MapDataset(Dataset):
    def __init__(
        self,
        data_dir: str | PathLike,
        data_df: pd.DataFrame,
        task: Literal["classification", "regression"],
        augment=False,
    ) -> None:
        self.data_dir = str(data_dir)
        self.df = data_df.copy()

        # check that the csv and the folder contain the same elements
        assert set(self.df["id"].values).issubset(set(os.listdir(data_dir)))
        self.task = task
        self.augment = augment

    def __len__(self) -> int:
        return self.df.__len__()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]["id"])
        img = decode_image(img_path, mode=ImageReadMode.GRAY)
        if self.augment:
            img = data_augmentation(img)

        label = self.__get_label(idx)
        
        return img, label

    def __get_label(self, idx: int) -> torch.Tensor:
        if self.task == "regression":
            label = self.df.iloc[idx]["area_perc"] / 100
            return torch.tensor(label, dtype=torch.float).unsqueeze(0)

        label = self.df.iloc[idx]["explored"]
        return torch.tensor((label, 1 - label), dtype=torch.float)


def get_loaders(batch_size: int, task: Literal['classification', 'regression']) -> tuple[DataLoader, DataLoader]:
    '''
        Returns a tuple containing the train dataloader and the validation dataloader (in this order)
    '''
    train_df = pd.read_csv(CSV_DATA_DIR / 'df_train.csv')
    val_df = pd.read_csv(CSV_DATA_DIR / 'df_valid.csv')

    train_ds = MapDataset(PROCESSED_DATA_DIR / "train", train_df, task, augment=True)
    val_ds = MapDataset(PROCESSED_DATA_DIR / "valid", val_df, task, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return train_loader, val_loader