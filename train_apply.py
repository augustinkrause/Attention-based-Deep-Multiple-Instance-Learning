import torch
import argparse
from data.data_utils.transformations import get_transformation
from data.data_utils.metrics import cv, nested_cv
from torch.utils.data import ConcatDataset
from data.data import load_data
from model.model_utils.setup_model_training import setup_model_training_cv
from training import train_single_epoch, evaluate, test


def train_apply(
		method = "attention", 
		dataset = "MNIST", 
		parameter_grid = {
			"n_epochs": [20],
			"learning_rate": [0.0005],
			"weight_decay": [0.0001],
			"momentum": [0.9], 
			"beta_1": [0.9],
			"beta_2": [0.999],
			"optimizer": ["Adam"],
			"mil_type": ["embeddings_based"]
		}):

	parameter_grid["pooling_type"] = [method]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Nested CV to get generalization estimate
	ds_train, ds_test = load_data(dataset, transformation=get_transformation(device, normalization_params=255. if dataset == "MNIST" else None))
	#ncv_error = nested_cv(ConcatDataset([ds_train, ds_test]), parameter_grid, dataset)

	# CV to get model params (only on training data, since we want to return the predicitons on test data, which should be unseen)
	params, _ = cv(ds_train, parameter_grid, dataset)
	print(f"CV found the following parameter combination: {params}")

	# get model predictions for the found model_params
	# set up training and load data
	model, optimizer, criterion = setup_model_training_cv(dataset, params)
	ds_train, ds_test = load_data(dataset, transformation=get_transformation(device, normalization_params=255. if dataset == "MNIST" else None))

	# train model
	model.train()
	for epoch in range(1, params["n_epochs"] + 1):
		train_single_epoch(model, ds_train, criterion, optimizer, 500, epoch)

	# test model
	y_pred, y_true, total_correct, total_samples = test(model, ds_test)
	#print(f"NCV estimated generalization error: {ncv_error}", flush=True)
	evaluate(y_pred, y_true, total_correct, total_samples)
	return y_pred # return only the predictions


def main():
	"""
        This can be called by running "python -m train_apply [PARAMETERS]
    """

	parser = argparse.ArgumentParser()
	parser.add_argument('--n-epochs', default= [20], nargs="+", type=int)
	parser.add_argument('--dataset', 
					 	default='MNIST', 
						choices=['MUSK1', 'MUSK2', 'ELEPHANT', 'TIGER', 'FOX', 'MNIST'])
	parser.add_argument('--mil-type', 
					 	default='embedding_based',
						choices=['embedding_based', 'instance_based'])
	parser.add_argument('--pooling-type', 
					 	default='max',
						choices=['max', 'mean', 'attention', 'gated_attention'])
	parser.add_argument('--n-train', default = None, type=int)
	parser.add_argument('--n-test', default = None, type=int)
	parser.add_argument('--learning-rate', default = [0.0005], nargs="+" , type=float)
	parser.add_argument('--weight-decay', default = [0.0001], nargs="+", type=float)
	parser.add_argument('--momentum', default = [0.9], nargs="+", type=float)
	parser.add_argument('--beta-1', default = [0.9], nargs="+", type=float)
	parser.add_argument('--beta-2', default = [0.999], nargs="+", type=float)
	parser.add_argument('--optimizer', default = ["Adam"], nargs="+", type=str)
	args = parser.parse_args()
	print("Args parsed!", flush=True)

	parameter_grid = {
		"n_epochs": args.n_epochs,
		"learning_rate": args.learning_rate,
		"weight_decay": args.weight_decay,
		"momentum": args.momentum, 
		"beta_1": args.beta_1,
		"beta_2": args.beta_2,
		"optimizer": args.optimizer,
		"mil_type": [args.mil_type]
	}
	y_pred = train_apply(args.pooling_type, args.dataset, parameter_grid)
	print(f"Predictions on test set: {y_pred}", flush=True)

if __name__ == '__main__':
	main()