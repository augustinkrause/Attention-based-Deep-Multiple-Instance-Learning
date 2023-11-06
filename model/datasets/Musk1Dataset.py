import os
import os.path as op
import PIL
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class Musk1Dataset(Dataset):
    """
    A class representing the MUSK1-dataset.
    """
    def __init__(
            self, dir: str, transform=None
    ) -> None:
        self.transform = transform
        mat = sio.loadmat(dir)
        self.bag_ids = np.array(mat['bag_ids'])[0]
        self.labels = np.array(mat['labels'].todense())[0]
        self.features = np.array(mat['features'].todense()) 
        print(self)

    def __len__(self):
        return np.unique(self.bag_ids).shape[0]
    
    def __str__(self):
        return f"Musk1Dataset:\n bag_ids:{self.bag_ids}\n labels:{self.labels}\n features:{self.features}"

    def __getitem__(self, i):
        # returns i-th bag
        feat = self.features[self.bag_ids == i]
        l = self.labels[self.bag_ids == i]
        return feat, l
