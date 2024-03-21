import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset

import os
import cv2
import pandas as pd
import numpy as np
from numpy import newaxis


class AudioImageDataset(Dataset):
    def __init__(self, file, audio_dir, image_dir):
        self.df = pd.read_csv(file)
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.img_transform = tv.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aud_transform = tv.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[-10.59], std=[85.66])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        aud_name = os.path.join(self.audio_dir, self.df.iloc[index, 0])
        audio = np.load(aud_name)
        # audio_wrk = torch.from_numpy(audio[newaxis, :, :]) without normalization, not preferred
        audio = self.aud_transform(audio[:, :, newaxis])
        audio[:, :, [0, 2]] = audio[:, :, [2, 0]]

        img_name = os.path.join(self.image_dir, self.df.iloc[index, 1])
        image = cv2.imread(img_name)
        image[:, :, [0, 2]] = image[:, :, [2, 0]]
        image = self.img_transform(image)

        sample = {'audio': audio, 'image': image, 'label': self.df.iloc[index, 2]}
        return sample


def get_loader(csv_fp, audio_dir, image_dir, batch_size, num_workers):
    ds = AudioImageDataset(file=csv_fp, audio_dir=audio_dir, image_dir=image_dir)
    if batch_size <= 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=num_workers)
    else:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader