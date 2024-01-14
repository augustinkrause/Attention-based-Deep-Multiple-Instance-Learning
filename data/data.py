import matplotlib.pyplot as plt
from data.datasets_loader.CLIMLIDataset import CL1MLIDataset
from data.datasets_loader.MNISTBagsDataset import MNISTBagDataset
import argparse
import math
import os
from pathlib import Path
from data.data_utils.transformations import get_transformation

def main():
    """
        This can be called by running "python -m data.data [PARAMETERS]
    """
    args = get_args()

    if(args.dataset != "MNIST"):
        raise ValueError('Dataset it not supported for visualization')
    
    dl_train, _ = load_data(args.dataset)

    data_to_visualize = [dl_train[i] for i in range(args.n_samples)] # get only the number of bags we want to visualize
    show(data_to_visualize, args.outfolder)


def load_data(dataset, transformation=None, n_train=None, n_test=None):

	
    if dataset == "MUSK1" or dataset == "MUSK2" or dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":
        dataloader_train = CL1MLIDataset(dataset, transformation, n_train, n_test, train = True)
        dataloader_test = CL1MLIDataset(dataset, transformation, n_train, n_test, train = False)
	    
    elif dataset == "MNIST":
        dataloader_train = MNISTBagDataset(transformation, n_train, n_test, train = True)
        dataloader_test = MNISTBagDataset(transformation, n_train, n_test, train = False)
    
    else: raise ValueError('Dataset it not supported')
    
    return  dataloader_train, dataloader_test



def visualize_bag_mnist(bag_data, label, outfolder=None, filename=None):


	
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
    if outfolder and filename:
        # create outfolder if not exist
        Path(outfolder).mkdir(parents=True, exist_ok=True)        
        plt.savefig(os.path.join(outfolder, filename))

def show(bags_data, outfolder = None):

	# bags must belong to MNIST

    for i, (bag, label) in enumerate(bags_data):
        if outfolder:
            visualize_bag_mnist(bag, label, outfolder, f'plot_{i}')
        else:
            visualize_bag_mnist(bag, label)



def get_args() -> argparse.Namespace:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--n-samples', default= 10, type=int)
    parser.add_argument('--outfolder', default= os.path.join(os.getcwd(), "data", "plots"))
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()

