import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os


def into_dict(index, features):

    dictionary = {}
    for index, value in zip(index, features):
        if index in dictionary:
            dictionary[index].append(value)
        else:
            dictionary[index] = [value]
    return dictionary


def load_data(dataset,
transformation=None,
n_train=None, n_test=None, ...):

	dataset = dataset.upper()
	 x_all = []
	 y_all = []

	if dataset == "MUSK1" or dataset == "MUSK2" or dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":

		 dataset = scipy.io.loadmat(os.path.join(os.getcwd() , "data" ,"datasets",f"{dataset}.mat" ))

		 instance_bag_ids = np.array(dataset['bag_ids'])[0]
		 instance_features = np.array(dataset['features'].todense()) 
		 instance_labels = np.array(dataset['labels'].todense())[0]

  		 bag_features = into_dictionary(instance_bag_ids,
                                       instance_features)
    	 bag_labels = into_dictionary(instance_bag_ids,
                                     instance_labels) 
         for i in range(min(instance_bag_ids), max(instance_bag_ids) + 1):

            x_all.append(np.array(bag_features.pop(i)))
            y_all.append(max(bag_labels[i]))



    elif dataset == "MNIST":

    	x_mnist, y_mnist = np.load(os.path.join(os.getcwd() , "data" ,"datasets","MNIST.npz" ))



    else:

    	raise ValueError('Dataset it not supported')

    if transformation:

    	x_all = [transformation(x) for x in x_all]


    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if n_train and n_test:

    	if n_train + n_test > len(x_all):
    		raise ValueError(f'Not enough data, max is f{len(x_all)}')


    	x_train = x_all[:n_train]
    	y_train = y_all[:n_train]
    	x_test =  x_all[n_train:n_train + n_test]
    	y_test =  y_all[n_train:n_train + n_test]



    return (((x,y) for x,y in zip(x_train,y_train)), (((x,y) for x,y in zip(x_test,y_test))


