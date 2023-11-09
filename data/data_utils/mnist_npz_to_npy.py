import torch.utils.data as data_utils
from torchvision import datasets, transforms
import os
import numpy as np

def main():
    """
    A script that stores a classical MIL dataset in .npy format.
    """

    mnist = np.load(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "mnist.npz"))
    np.save(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_features.npy"), mnist["arr_0"], allow_pickle=False)
    np.save(os.path.join(os.getcwd(), "data", "datasets", "MNIST", "MNIST_labels.npy"), mnist["arr_1"], allow_pickle=False)

if __name__ == '__main__':
    main()