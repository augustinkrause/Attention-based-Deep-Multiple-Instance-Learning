import os
from torch.utils.data import Dataset
import numpy as np
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
        self.n_train = n_train
        self.n_test = n_test

        if self.n_train is None or self.n_test is None:
            create_mnist(target_number = self.target_number)

        else:
            create_mnist(target_number = self.target_number, num_bag_train = self.n_train,num_bag_test = self.n_test )


        self.bag_ids_train = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids_train.npz"))
        self.bag_ids_test = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_bag_ids_test.npz"))


        self.n_train = len(self.bag_ids_train.files) 
        self.n_test = len(self.bag_ids_test.files) 


        # lazily loading data
        self.features = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_features.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_labels.npy"), mmap_mode='r')

        self.bag_ids_train_ids = np.unique(self.bag_ids_train.files)
        random.Random(seed).shuffle(self.bag_ids_train_ids)
        self.bag_ids_test_ids = np.unique(self.bag_ids_test.files)
        random.Random(seed).shuffle(self.bag_ids_test_ids)

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, i):


        if self.train:

            bag_map = self.bag_ids_train_ids
            bag_ids = self.bag_ids_train
        else:
            bag_map = self.bag_ids_test_ids
            bag_ids = self.bag_ids_test

        # return the ith bag
        if self.transformation:
            bag_ids_i = bag_ids[bag_map[i]]
            label = 1 if self.target_number in self.labels[bag_ids_i] else 0
            return self.transformation(np.array([ feat.reshape(1,28,28) for feat in self.features[bag_ids_i] ]), np.array([label]))

        else:
            bag_ids_i = bag_ids[bag_map[i]]
            label = 1 if self.target_number in self.labels[bag_ids_i] else 0
            return np.array([feat.reshape(1,28,28) for feat in self.features[bag_ids_i]]) , np.array([label])