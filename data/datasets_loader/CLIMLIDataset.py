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
        self.n_bags = np.unique(self.bag_ids).shape[0]
        self.train = train

        if n_train is None or n_test is None:
            n_train = int(0.8 * self.n_bags)
            n_test = self.n_bags - n_train
        elif n_train + n_test > self.n_bags:
            raise ValueError(f'Not enough data for desired train/test split, max is {self.n_bags}')

        self.n_train = n_train
        self.n_test = n_test

        
        # shuffling
        unique_bag_ids = np.unique(self.bag_ids)
        random.Random(seed).shuffle(unique_bag_ids)

        self.bag_ids_train = unique_bag_ids[0:self.n_train]
        self.bag_ids_test = unique_bag_ids[self.n_train:self.n_train + self.n_test]

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test
        
    def __str__(self):
        if self.train:
            return f" bag_ids_train: {self.bag_ids_train}\n n_train: {self.n_train}\n n_bags:{self.n_bags}\n"
        else :
            return f" bag_ids_test: {self.bag_ids_test}\n n_test: {self.n_test}\n n_bags:{self.n_bags}\n"

    def __getitem__(self, i):
        # Return i-th bag


        if self.train:

            bag_map = self.bag_ids_train
        else:
            bag_map = self.bag_ids_test

        if self.transformation:
            return self.transformation(self.features[self.bag_ids == bag_map[i]]), np.max(self.labels[self.bag_ids == bag_map[i]])
        else:
            return self.features[self.bag_ids == bag_map[i]], np.max(self.labels[self.bag_ids == bag_map[i]])

