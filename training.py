import argparse
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from data.data import load_data
from model.model_utils.setup_model_training import setup_model_training, setup_model_training_cv
from data.data_utils.transformations import get_transformation
import os
from pathlib import Path

def train(model, ds, n_epochs, criterion, optimizer, print_freq):

	print("Beginning training", flush=True)
	model.train()
    
	for epoch in range(1, n_epochs +1):
		train_single_epoch(model, ds, criterion, optimizer, print_freq, epoch)
	
def train_single_epoch(model, ds, criterion, optimizer, print_freq, epoch):
	#print(f"Beginning epoch {epoch}", flush=True)

	total_loss = 0.0
	counter = 0
	right_preds = 0
	try:
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
				print(f"Epoch {epoch}, Bag number {counter}, Training Loss {total_loss/counter:.5f}, Training Accuracy = {right_preds/counter:.5f}", flush=True)
	except Exception as e:
		print(e, flush=True)


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

	print(f"Test Accuracy: {accuracy:.5f}", flush=True)
	print(f"precision: {precision_score(y_true, y_pred):.5f}", flush=True)
	print(f"recall: {recall_score(y_true, y_pred):.5f}", flush=True)
	print(f"F-Score: {f1_score(y_true, y_pred):.5f}", flush=True)
	print(f"AUC: {roc_auc_score(y_true, y_pred):.5f}", flush=True)

	return accuracy, precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred), roc_auc_score(y_true, y_pred)

def main():
	"""
        This can be called by running "python -m training [PARAMETERS]
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-epochs', default= 20, type=int)
	parser.add_argument('--dataset', 
					 	default='MNIST', 
						choices=['MUSK1', 'MUSK2', 'ELEPHANT', 'TIGER', 'FOX', 'MNIST', 'TCGA'])
	parser.add_argument('--mil-type', 
					 	default='embedding_based',
						choices=['embedding_based', 'instance_based'])
	parser.add_argument('--pooling-type', 
					 	default='max',
						choices=['max', 'mean', 'attention', 'gated_attention'])
	parser.add_argument('--n-train', default = None, type=int)
	parser.add_argument('--n-test', default = None, type=int)
	parser.add_argument('--learning-rate', default = 0.0005 , type=float)
	parser.add_argument('--weight-decay', default = 0.0001, type=float)
	parser.add_argument('--momentum', default = 0.9, type=float)
	parser.add_argument('--beta-1', default = 0.9, type=float)
	parser.add_argument('--beta-2', default = 0.999, type=float)
	parser.add_argument('--print-freq', default=100, type=int)
	parser.add_argument('--parameter-path', default=None, type=str)
	parser.add_argument('--optimizer', 
						default = "Adam", 
						choices = ["Adam", "SGD"],
						type=str)
	args = parser.parse_args()
	print("Args parsed!", flush=True)
	print(f"Args: {args}", flush=True)

	if args.parameter_path == None:
		model, device, optimizer, criterion, transformation = setup_model_training(args)
		model.to(device)
	else:
		model, optimizer, criterion = setup_model_training_cv(args.dataset, vars(args))
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		transformation = get_transformation(device, normalization_params=255. if args.dataset == "MNIST" else None)
	ds_train, ds_test = load_data(args.dataset, transformation=transformation, n_train = args.n_train, n_test = args.n_test)

	if args.parameter_path == None:
		train(model, ds_train, args.n_epochs, criterion, optimizer, args.print_freq)
		y_pred, y_true, total_correct, total_samples = test(model, ds_test)
		evaluate(y_pred, y_true, total_correct, total_samples)
	else:
		train(model, ds_train, args.n_epochs, criterion, optimizer, args.print_freq)
		Path(args.parameter_path).mkdir(parents=True, exist_ok=True)
		torch.save(model.state_dict(), os.path.join(args.parameter_path, f"{args.dataset}_{args.mil_type}_{args.pooling_type}.pt"))

if __name__ == '__main__':
	main()