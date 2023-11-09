import os
import os.path as op
import PIL
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MNISTBagDataset(Dataset):
    """
    A class representing a data set of MNIST-bags.
    """
    def __init__(
            self, transformation=None, n_train=None, n_test=None, train=True, seed=42, target_number=9
    ) -> None:
        self.transformation = transformation
        self.seed = seed
        self.train = train
        self.target_number = target_number

        self.bag_ids_train = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids_train.npz"))
        self.bag_ids_test = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids_test.npz"))
        self.n_bags = len(self.bag_ids_train.files) + len(self.bag_ids_test.files)

        self.r = np.random.RandomState(seed)

        if n_train is None and n_test is None:
            n_train = int(0.8 * self.n_bags)
            n_test = self.n_bags - n_train
        elif n_train + n_test > self.n_bags:
            raise ValueError(f'Not enough data for desired train/test split, max is {self.n_bags}')
        self.n_train = n_train
        self.n_test = n_test

        # lazily loading data
        self.features = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_features.npy"))
        self.labels = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_labels.npy"))

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, i):
        # return the ith bag
        if self.train:
            bag_ids_i = self.bag_ids_train[f"{i}"]
        else:
            bag_ids_i = self.bag_ids_test[f"{i}"]
        label = 1 if self.target_number in self.labels[bag_ids_i] else 0
        return self.features[bag_ids_i], label
