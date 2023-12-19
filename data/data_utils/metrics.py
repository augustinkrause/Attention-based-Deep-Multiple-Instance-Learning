import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist  # fast distance matrices
import math
from model.model_utils.setup_model_training import setup_model_training_cv
from training import train, test
from data.data_utils.transformations import get_transformation
from data.data import load_data
import torch

def auc(y_true, y_pred, plot=False):
	"""
    calculates the area under the ROC the (receiver operating curve) graph of the given data

    :y_ture: true predicition values {-1,1}
    :y_pred: predicted values between -1 and 1
    :plot: plots the ROC graph if true
    :return: area under the ROC graph
    """ 

	o = y_pred.argsort()[::-1]

	pos_len = len([y for y in y_true if y == 1])
	neg_len = len([y for y in y_true if y == 0])
	pos_step = 1/pos_len
	neg_step = 1/neg_len



	n = len(y_true)
	fpr = np.zeros(n + 1) 
	tpr = np.zeros(n + 1) 

	auc = 0
	for i in range(1, n+1):
		fpr[i] = fpr[i-1] + (1/2)*(1 - y_true[o[i-1]]) * neg_step
		tpr[i] = tpr[i-1] + (1/2)*(1 + y_true[o[i-1]]) * pos_step

		auc += (fpr[i] - fpr[i-1])* (tpr[i])

	if plot:
		plt.title('Receiver Operating Characteristic (ROC)')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()

	return auc



def zero_one_loss(y_true, y_pred):
    ''' 
    Computes the 0-1 loss which measures the average number of falsely predicted labels.
    
    :y_true: True prediction values {0, 1}
    :y_pred: Predicted values
    :return: The computed 0-1 loss
    ''' 
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same.")

    # Count the number of mismatches
    num_mismatches = sum(1 for true, pred in zip(y_true, y_pred) if true != pred)

    # Calculate the average 0-1 loss
    total_samples = len(y_true)
    zero_one_loss = num_mismatches / total_samples

    return zero_one_loss



def chunks(arr, n_parts):

    ''' 
    divides the given array into n_parts equal partitions, if that is not possible the last partition will not be equal to the rest

    :arr: array to be partitioned
    :n_parts: number of partitions 
    :return: list of divided partitions

    '''

    l_chunks = []

    k = math.ceil(len(arr)/n_parts)
    for i in range(0,len(arr),k):

        if i + k > len(arr):
            l_chunks.append(arr[i : len(arr)])
        else:
            l_chunks.append(arr[i: i + k])


    return l_chunks


def cv(ds, params, dataset, loss_function=zero_one_loss, nfolds=10, nrepetitions=1, n_epochs = 20, print_freq = 100):

    ''' 
    computes the n-fold cross-validation on every combination of the given parameters using the given loss function
    and then returns the class with the least cross-validation loss and saves the loss as an attribute in it

    :ds: dataset
    :loss_function: a function handle to the loss function to be used. It should have the following
        signature:
        l = loss_function(y_true, y_pred)
        where y_true are the true targets y and y_pred are the predicted targets yb. This parameter is optional
        with the standard value mean_absolute_error

    :nfolds: the number of partitions (m in the guide). This parameter should be optional with a standard value of 10.
    :nrepetitions: the number of repetitions (r in the guide). This parameter is optional with the standard value 5.
    :return: the best params and the minimum cv error
    '''

    min_error = np.inf 
    min_param = None
    min_class = None

    number_params = len(list(it.product(*list(params.values()))))
    for param_idx , param in enumerate(list(it.product(*list(params.values())))):

        param_dict = dict(zip(params.keys(), param))
        print(f"Testing for parameter combination {param_dict} - {param_idx+1}/{number_params}\n")
        error = 0

        for r in range(nrepetitions):

            print(f"In repetition number {r+1}/{nrepetitions}\n")

            ds_folded = chunks(ds, nfolds)

            for part_idx in range(nfolds):

                print(f"In partition number {part_idx+1}/{nfolds}")

                training_ds = ds_folded.copy()
                del training_ds[part_idx]
                training_ds = [element for sublist in training_ds for element in sublist]

                testing_ds = ds_folded[part_idx].copy()
                
                model, optimizer, criterion = setup_model_training_cv(dataset, param_dict)
                train(model, training_ds, n_epochs, criterion, optimizer, print_freq)
                y_pred, y_true, total_correct, total_samples = test(model, testing_ds)
                error += loss_function(y_true, y_pred)

        error = error / (nrepetitions * nfolds)

        if error < min_error:
            min_error = error
            min_param = param_dict
  
    return min_param, min_error


def nested_cv(ds, params, dataset, loss_function=zero_one_loss, outer_nfolds=10, inner_nfolds = 5, nrepetitions=1, n_epochs = 20, print_freq = 100):

	
    min_error = np.inf 
    min_param = None
    min_class = None

    ds_folded = chunks(ds, outer_nfolds)

    for part_idx in range(outer_nfolds):

        print(f"In outer partition number {part_idx+1}/{outer_nfolds}")

        training_ds = ds_folded.copy()
        del training_ds[part_idx]
        training_ds = [element for sublist in training_ds for element in sublist]

        testing_ds = ds_folded[part_idx].copy()
        
        param_dict, error = cv(training_ds, params, "MUSK1", loss_function=loss_function, 
        	nfolds = inner_nfolds, nrepetitions=nrepetitions, n_epochs = n_epochs, print_freq = print_freq)

        model, optimizer, criterion = setup_model_training_cv(dataset, param_dict)
        train(model, training_ds, n_epochs, criterion, optimizer, print_freq)
        y_pred, y_true, total_correct, total_samples = test(model, testing_ds)
        error = loss_function(y_true, y_pred)

        if error < min_error:
            min_error = error
            min_param = param_dict

    return min_param, min_error


def main():
	"""
        Test cross-validation
    """

	params = {
	    'mil_type': ['embedding_based'],
	    'pooling_type': ['max', "mean", "attention", "gated_attention"],
	    'learning_rate': [0.0005],
	    'weight_decay': [0.0001],
	    'momentum': [0.9],
	    'beta_1': [0.9],
	    'beta_2': [0.999],
	    "optimizer": ["Adam", "SGD"]
	}


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ds_train, ds_test = load_data("MUSK1", transformation=get_transformation(device))

	ds_l = [t for t in ds_train]
	param, error = nested_cv(ds_l, params, "MUSK1")

	print("Best params: ",param)
	print("Minimum CV error: ", error)

if __name__ == '__main__':
	main()
