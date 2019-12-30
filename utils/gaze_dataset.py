import torch
from torch.utils.data import Dataset
import numpy as np


class GazeDataset(Dataset):
    def __init__(self, filepath, transform=None):
        assert filepath.is_file() or filepath.isfile(), 'specify path to npz file'
        self.data = np.load(filepath, mmap_mode='r+')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = self.data[item]

        if self.transform:
            image = self.transform(image)
        return image
