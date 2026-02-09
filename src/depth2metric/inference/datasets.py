import os

import cv2
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from torchvision.io.image import decode_image


class NYUDataset(Dataset):

    def __init__(self, data_path, transforms=None, load_with_opencv=False):
        super().__init__()

        csv_path = os.path.join(data_path, "nyu2_train.csv")
        self.df = pd.read_csv(
            csv_path, names=["color_image", "depth_image"]
        ).map(lambda s: os.path.join(data_path, s[5:]))

        self.load_with_opencv = load_with_opencv

        if transforms is None:
            self.transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        color_path, depth_path = self.df.iloc[index].to_list()

        if self.load_with_opencv:
            color_image = cv2.imread(color_path, cv2.IMREAD_COLOR_RGB)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_COLOR_RGB)
        else:
            color_image = decode_image(color_path)
            depth_image = decode_image(depth_path)

        tr_color_image, tr_depth_image = self.transforms(color_image), self.transforms(depth_image)

        return color_image, tr_color_image, depth_image, tr_depth_image
