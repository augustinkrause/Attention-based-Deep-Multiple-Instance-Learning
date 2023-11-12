import os
import os.path as op
import PIL
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random
from data.data_utils.create_mnist_bags import create_mnist 

class MNISTBagDataset(Dataset):
    """
    A class representing a data set of MNIST-bags.
    """
    def __init__(
            self, transformation=None, n_train=None, n_test=None, train=True, seed=42, target_number=9,
    ) -> None:
        self.transformation = transformation
        self.seed = seed
        self.train = train
        self.target_number = target_number
        create_mnist(target_number = self.target_number)
        self.bag_ids = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids.npz"))
        self.n_bags = len(self.bag_ids.files) 


        if n_train is None or n_test is None:
            n_train = int(0.8 * self.n_bags)
            n_test = self.n_bags - n_train
        elif n_train + n_test > self.n_bags:
            raise ValueError(f'Not enough data for desired train/test split, max is {self.n_bags}')
        self.n_train = n_train
        self.n_test = n_test

        # lazily loading data
        self.features = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_features.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_labels.npy"), mmap_mode='r')

        unique_bag_ids = np.unique(self.bag_ids.files)
        random.Random(seed).shuffle(unique_bag_ids)

        self.bag_ids_train = unique_bag_ids[0:self.n_train]
        self.bag_ids_test = unique_bag_ids[self.n_train:self.n_train + self.n_test]

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, i):


        if self.train:

            bag_map = self.bag_ids_train
        else:
            bag_map = self.bag_ids_test

        # return the ith bag
        if self.transformation:
            bag_ids_i = self.bag_ids[f"{bag_map[i]}"]
            label = 1 if self.target_number in self.labels[bag_ids_i] else 0
            return self.transformation(np.array([ feat.reshape(1,28,28) for feat in self.features[bag_ids_i] ])), label

        else:
            bag_ids_i = self.bag_ids[f"{bag_map[i]}"]
            label = 1 if self.target_number in self.labels[bag_ids_i] else 0
            return np.array([feat.reshape(1,28,28) for feat in self.features[bag_ids_i]]) , label
        

        return self.features[bag_ids_i], label

cd = MNISTBagDataset()
