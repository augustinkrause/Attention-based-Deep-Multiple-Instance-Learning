import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from datasets_loader.CLIMLIDataset import CL1MLIDataset
from datasets_loader.MNISTBagsDataset import MNISTBagDataset
import os
import argparse

def main():
    """

    """
    args = get_args()
    
    dl_train, dl_test = load_data(args.dataset, n_train=args.n_train, n_test=args.n_test)
    print(dl_train.__getitem__(10))
    print(dl_test.__getitem__(10))


def load_data(dataset, transformation=None, n_train=None, n_test=None):

	
    if dataset == "MUSK1" or dataset == "MUSK2" or dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":
        dataloader_train = CL1MLIDataset(dataset, transformation, n_train, n_test, train = True)
        dataloader_test = CL1MLIDataset(dataset, transformation, n_train, n_test, train = False)
	    
    elif dataset == "MNIST":
        dataloader_train = MNISTBagDataset(transformation, n_train, n_test, train = True)
        dataloader_test = MNISTBagDataset(transformation, n_train, n_test, train = False)
    
    else: raise ValueError('Dataset it not supported')
    
    return  dataloader_train, dataloader_test


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MUSK1')
    parser.add_argument('--n-train', type=int)
    parser.add_argument('--n-test', type=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()

