import os
import os.path as op
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.Musk1Dataset import Musk1Dataset
import argparse
import numpy as np


def main():
    """

    """
    args = get_args()
    
    load_data(args.dataset)


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='musk1')
    args = parser.parse_args()

    return args

def load_data(dataset):
    if(dataset == "musk1"):
        ds = Musk1Dataset("../data/datasets/MUSK1/MUSK1.mat")
        print(len(ds))
        feat1, l1 = ds.__getitem__(1)
        print(feat1.shape)
        print(l1.shape)
    


if __name__ == '__main__':
    main()