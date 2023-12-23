import argparse
import torch
from data.data import load_data
from model.model_utils.setup_model_training import setup_model_training
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from data.data_utils.transformations import get_transformation
from data.data_utils.metrics import cv, nested_cv
from torch.utils.data import ConcatDataset

def train(model, ds, n_epochs, criterion, optimizer, print_freq):

	print("Beginning training")
	model.train()
    
	for epoch in range(1, n_epochs +1):
		train_single_epoch(model, ds, criterion, optimizer, print_freq, epoch)

def train_apply(
		method = "attention", 
		dataset = "MNIST", 
		parameter_grid = {
			"n_epochs": [20],
			"learning_rate": [0.0005],
			"weight_decay": [0.0001],
			"momentum": [0.9], 
			"beta_1": 0.9,
			"beta_2": 0.999,
			"optimizer": ["Adam"],
			"mil_type": ["embeddings_based"]
		}):

	parameter_grid["pooling_type"] = [method]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Nested CV to get generalization estimate
	ds_train, ds_test = load_data(dataset, transformation=get_transformation(device))
	ncv_error = nested_cv(ConcatDataset([ds_train, ds_test]), parameter_grid, dataset)

	# CV to get model params (only on training data, since we want to return the predicitons on test data, which should be unseen)
	params = cv(ds_train, parameter_grid, dataset)

	# get model predictions for the found model_params
	params["dataset"] = dataset
	args = argparse.Namespace(**params)

	# set up training and load data
	model, device, optimizer, criterion, transformation = setup_model_training(args)
	model.to(device)
	#ds_train, ds_test = load_data(dataset, transformation=transformation)

	# train model
	model.train()
	for epoch in range(1, params["n_epochs"] + 1):
		train_single_epoch(model, ds_train, criterion, optimizer, 100, epoch)

	# test model
	return test(model, ds_test)[0] # return only the predictions
		
def train_single_epoch(model, ds, criterion, optimizer, print_freq, epoch):
	print(f"Beginning epoch {epoch}")

	total_loss = 0.0
	counter = 0
	right_preds = 0
	for bag, label in ds:

		optimizer.zero_grad()
		output = model(bag)
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		pred = float((output.item() >= 0.5))
		right_preds += (pred == label.item()) 

		counter += 1

		if counter % print_freq == 0:
			print(f"Epoch {epoch}, Bag number {counter}, Training Loss {total_loss/counter:.5f}, Training Accuracy = {right_preds/counter:.5f}")


def test(model, ds):
	model.eval()

	total_correct = 0
	total_samples = 0

	total_pred = []
	total_true = []
	with torch.no_grad():
		for bag, label in ds:
			output = model(bag)
			pred = float((output.item() >= 0.5))
			total_correct += (pred == label.item())
			total_samples += 1
			total_pred.append(pred)
			total_true.append(label.cpu().item())

	y_pred = np.array(total_pred)
	y_true = np.array(total_true)
	return y_pred, y_true, total_correct, total_samples
	
def evaluate(y_pred, y_true, total_correct, total_samples):
	accuracy = total_correct / total_samples

	print(f"Test Accuracy: {accuracy:.5f}")
	print(f"precision: {precision_score(y_true, y_pred):.5f}")
	print(f"recall: {recall_score(y_true, y_pred):.5f}")
	print(f"F-Score: {f1_score(y_true, y_pred):.5f}")
	print(f"AUC: {roc_auc_score(y_true, y_pred):.5f}")

	return accuracy, precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred), roc_auc_score(y_true, y_pred)

def main():
	"""
        This can be called by running "python -m training [PARAMETERS]
    """

	parser = argparse.ArgumentParser()
	parser.add_argument('--n-epochs', default= 20, type=int)
	parser.add_argument('--dataset', 
					 	default='MNIST', 
						choices=['MUSK1', 'MUSK2', 'ELEPHANT', 'TIGER', 'FOX', 'MNIST'])
	parser.add_argument('--mil-type', 
					 	default='embedding_based',
						choices=['embedding_based', 'instance_based'])
	parser.add_argument('--pooling-type', 
					 	default='max',
						choices=['max', 'mean', 'attention', 'gated_attention'])
	parser.add_argument('--n-train', default = 1000, type=int)
	parser.add_argument('--n-test', default = 1000, type=int)
	parser.add_argument('--learning-rate', default = 0.0005 , type=float)
	parser.add_argument('--weight-decay', default = 0.0001, type=float)
	parser.add_argument('--momentum', default = 0.9, type=float)
	parser.add_argument('--beta-1', default = 0.9, type=float)
	parser.add_argument('--beta-2', default = 0.999, type=float)
	parser.add_argument('--print-freq', default=100, type=int)
	args = parser.parse_args()
	print("Args parsed!")

	model, device, optimizer, criterion, transformation = setup_model_training(args)
	model.to(device)

	ds_train, ds_test = load_data(args.dataset, transformation=transformation, n_train = args.n_train, n_test = args.n_test)

	train(model, ds_train, args.n_epochs, criterion, optimizer, args.print_freq)
	y_pred, y_true, total_correct, total_samples = test(model, ds_test)
	evaluate(y_pred, y_true, total_correct, total_samples)

if __name__ == '__main__':
	main()