import os
import os.path as op
import argparse
import numpy as np
import scipy.io as sio


def main():
    """
    A script that stores a classical MIL dataset in .npy format.
    """
    args = get_args()
    
    convert_dataset(args.dataset)


def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MUSK1')
    args = parser.parse_args()

    return args

def convert_dataset(dataset):
    mat = sio.loadmat(os.path.join(os.getcwd(), "data", "datasets", f"{dataset}", f"{dataset}.mat"))
    bag_ids = np.array(mat['bag_ids'])[0]
    features = np.array(mat['features'].todense())
    labels = np.array(mat['labels'].todense())[0]
    np.save(os.path.join(os.getcwd(), "data", "datasets", f"{dataset}", f"{dataset}_bag_ids.npy"), bag_ids, allow_pickle=False)
    np.save(os.path.join(os.getcwd(), "data", "datasets", f"{dataset}", f"{dataset}_labels.npy"), labels, allow_pickle=False)
    np.save(os.path.join(os.getcwd(), "data", "datasets", f"{dataset}", f"{dataset}_features.npy"), features, allow_pickle=False)
    


if __name__ == '__main__':
    main()