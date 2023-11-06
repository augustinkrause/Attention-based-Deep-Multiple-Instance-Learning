import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from data.datasets_loader.CLIMLIDataset import CL1MLIDataset
import os

def load_data(dataset, transformation=None, n_train=None, n_test=None):

	
    if dataset == "MUSK1" or dataset == "MUSK2" or dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":
        dataloader_train = CL1MLIDataset(dataset, transformation, n_train, n_test, train = True)
        dataloader_test = CL1MLIDataset(dataset, transformation, n_train, n_test, train = False)
	    
    elif dataset == "MNIST": pass
    
    else: raise ValueError('Dataset it not supported')
    
    return  dataloader_train, dataloader_test

