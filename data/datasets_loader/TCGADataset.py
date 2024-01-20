import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import random

class TCGADataset(Dataset):
    def __init__(self, n_train=None, n_test=None, train = True, seed = 42, shuffle = False):

        self.shuffle = shuffle
        self.seed = seed
        
        # load metadata (for labels)
        self.case_metadata = pd.read_csv(os.path.join(os.getcwd(), "data", "datasets", "TCGA", "case_metadata.csv"))
        self.slide_metadata = pd.read_csv(os.path.join(os.getcwd(), "data", "datasets", "TCGA", "slide_metadata.csv"))
        cols = ["case_id", "slide_id", "ajcc_pathologic_t"]
        self.metadata = pd.merge(self.case_metadata, self.slide_metadata, how="inner")[cols]

        # store slides paths
        self.slides = {slide_id : os.path.join(os.getcwd(), "data", "datasets", "TCGA", f"{slide_id}.npz") for slide_id in self.metadata["slide_id"]}
        self.n_bags = len(self.metadata)
        self.train = train

        if n_train is None or n_test is None:
            n_train = int(np.floor(0.8 * self.n_bags))
            n_test = self.n_bags - n_train
            # TODO: do we need to take care of the "overlapping" cases?
        elif n_train + n_test > self.n_bags:
            raise ValueError(f'Not enough data for desired train/test split, max is {self.n_bags}')

        self.n_train = n_train
        self.n_test = n_test

        
        # shuffling
        self.order = np.arange(len(self.metadata))
        if self.shuffle:
            random.Random(self.seed).shuffle(self.order)

        self.metadata_train = self.metadata.loc[self.order[:self.n_train]]
        self.metadata_test = self.metadata.loc[self.order[self.n_train:self.n_train + self.n_test]]

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test
        
    def __str__(self):
        if self.train:
            return f" metadata_train: {self.metadata_train}\n n_train: {self.n_train}\n n_bags:{self.n_bags}\n"
        else :
            return f" metadata_test: {self.metadata_test}\n n_test: {self.n_test}\n n_bags:{self.n_bags}\n"

    def _get_bag(self, bag_path : str):
        bag_npz = np.load(bag_path)
        bag = np.zeros((len(bag_npz.files), 768))
        for i, patch in enumerate(bag_npz.files):
            bag[i] = bag_npz[patch][0]
        print(bag.shape)
            
        return bag

    def _get_positive_bag_proportion(self):
        if self.train:
            sumt4_col = self.metadata_train["ajcc_pathologic_t"].apply(lambda x: 1 if x.startswith("T4") else 0)
            return sumt4_col.sum()/self.n_train
        else:
            sumt4_col = self.metadata_test["ajcc_pathologic_t"].apply(lambda x: 1 if x.startswith("T4") else 0)
            return sumt4_col.sum()/self.n_test

    def __getitem__(self, i):
        # Return i-th bag

        if self.train:
            bag_path = self.slides[self.metadata_train.loc[self.order[i]]["slide_id"]]
            label = np.array([1]) if self.metadata_train.loc[self.order[i]]["ajcc_pathologic_t"].startswith("T4") else np.array([0])
        else:
            bag_path = self.slides[self.metadata_test.loc[self.order[self.n_train + i]]["slide_id"]]
            label = np.array([1]) if self.metadata_test.loc[self.order[self.n_train + i]]["ajcc_pathologic_t"].startswith("T4") else np.array([0])
        bag = self._get_bag(bag_path)
        
        return bag, label

