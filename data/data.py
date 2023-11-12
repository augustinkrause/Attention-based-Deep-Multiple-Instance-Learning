import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from data.datasets_loader.CLIMLIDataset import CL1MLIDataset
from data.datasets_loader.MNISTBagsDataset import MNISTBagDataset
import os
import argparse
import math

def main():
    """

    """
    args = get_args()
    
    dl_train, dl_test = load_data(args.dataset, n_train=args.n_train, n_test=args.n_test)
    # print(dl_train.__getitem__(10))
    # print(dl_test.__getitem__(10))
    visualize_data(args.dataset)


def load_data(dataset, transformation=None, n_train=None, n_test=None):

	
    if dataset == "MUSK1" or dataset == "MUSK2" or dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":
        dataloader_train = CL1MLIDataset(dataset, transformation, n_train, n_test, train = True)
        dataloader_test = CL1MLIDataset(dataset, transformation, n_train, n_test, train = False)
	    
    elif dataset == "MNIST":
        dataloader_train = MNISTBagDataset(transformation, n_train, n_test, train = True)
        dataloader_test = MNISTBagDataset(transformation, n_train, n_test, train = False)
    
    else: raise ValueError('Dataset it not supported')
    
    return  dataloader_train, dataloader_test



def visualize_bag_mnist(bag_data, label):


	
    num_images = len(bag_data)
    num_columns = 5 
    num_rows = math.ceil(num_images  / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 2 * num_rows))
    axes = axes.reshape(num_rows, num_columns)
    fig.suptitle(f'Label: {label} - Number of instances: {num_images}')


    for i, image in enumerate(bag_data):


        ax = axes[i // num_columns, i % num_columns]
        ax.imshow(image.reshape(28,28), cmap='gray')
        ax.axis('off')
        

  	# Remove empty subplots if necessary
    for i in range(num_images, num_rows * num_columns):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

def visualize_data(dataset, num_samples = 10):

	if dataset == "MNIST":

		counter = 0
		generatror = load_data(dataset)[0]

		for bag, label in generatror:
			visualize_bag_mnist(bag, label)
			counter += 1
			if(counter == num_samples):
				break



	else:

		raise ValueError('Dataset it not supported')

def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--n-train', type=int)
    parser.add_argument('--n-test', type=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()

