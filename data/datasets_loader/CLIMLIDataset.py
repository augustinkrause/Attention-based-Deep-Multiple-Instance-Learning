import os
from torch.utils.data import Dataset
import numpy as np
import random

class CL1MLIDataset(Dataset):
    def __init__(self, dataset, transformation=None, n_train=None, n_test=None, train = True, seed = 42):

        if dataset != "MUSK1" and dataset != "MUSK2" and dataset != "FOX" and dataset != "ELEPHANT" and dataset != "TIGER":
            raise ValueError('Datset is not a classical MLI dataset')

        self.transformation = transformation
        self.bag_ids = np.load(os.path.join(os.getcwd(), "data", "datasets", f"{dataset}", f"{dataset}_bag_ids.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(os.getcwd(), "data", "datasets", f"{dataset}", f"{dataset}_labels.npy"), mmap_mode='r')
        self.features = np.load(os.path.join(os.getcwd(), "data", "datasets",f"{dataset}", f"{dataset}_features.npy"), mmap_mode='r')
        self.n_bags = max(self.bag_ids)
        self.train = train

        if n_train is None and n_test is None:
            n_train = int(0.8 * self.n_bags)
            n_test = self.n_bags - n_train
        elif n_train + n_test > self.n_bags:
            raise ValueError(f'Not enough data, max is {self.n_bags}')

        self.n_train = n_train
        self.n_test = n_test

        bag_map = {}
        for idx, bag_id in enumerate(self.bag_ids):
            bag_map.setdefault(bag_id, []).append(idx)

        self.bag_idx_l_train = list(bag_map.values())[:n_train]
        self.bag_idx_l_test = list(bag_map.values())[n_train:]

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, i):
        # Return i-th bag


        if self.train:

            bag_map = self.bag_idx_l_train
        else:
            bag_map = self.bag_idx_l_test

        if self.transformation:
            return [self.transformation(feat) for feat in self.features[bag_map[i][0]:bag_map[i][-1] + 1]], max(self.labels[bag_map[i][0]:bag_map[i][-1] + 1])
        else:
            return self.features[bag_map[i][0]:bag_map[i][-1] + 1], max(self.labels[bag_map[i][0]:bag_map[i][-1] + 1])

