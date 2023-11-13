import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.data import load_data
from model.model import MILModel


def train(model, ds, n_epochs, criterion, optimizer, print_freq):

	model.train()
    
	for epoch in range(1, n_epochs +1):
		print(f"Beginning epoch {epoch}")

		total_loss = 0.0
		counter = 0
		right_preds = 0
		for bag, label in ds:

			optimizer.zero_grad()
			output = model(bag)
			loss = criterion(output.squeeze(), torch.tensor(label))
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

    with torch.no_grad():
        for bag, label in ds:
            output = model(bag)
            pred = float((output.item() >= 0.5))
            total_correct += (pred == label.item())
            total_samples += 1

    accuracy = total_correct / total_samples

    print(f"Test Accuracy: {accuracy:.5f}")
    return accuracy


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--n-epochs', default= 10, type=int)
	parser.add_argument('--dataset', default='MUSK2')
	parser.add_argument('--mil_type', default='embedding_based')
	parser.add_argument('--pooling_type', default='gated attention')
	parser.add_argument('--n-train', default = 70, type=int)
	parser.add_argument('--n-test', default = 20, type=int)
	parser.add_argument('--learning-rate', default = 0.0005 , type=float)
	parser.add_argument('--weight-decay', default = 0.0001, type=int)
	parser.add_argument('--beta-1', default = 0.9, type=int)
	parser.add_argument('--beta-2', default = 0.999, type=int)
	parser.add_argument('--print-freq', default=100, type=int)
	args = parser.parse_args()

	model = MILModel(args.dataset, args.mil_type, args.pooling_type )
	model.double()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
		weight_decay=args.weight_decay, betas=(args.beta_1, args.beta_2))
	criterion = nn.BCELoss()

	ds_train, ds_test = load_data(args.dataset, n_train = args.n_train, n_test = args.n_test)

	train(model, ds_train, args.n_epochs, criterion, optimizer, args.print_freq)
	test(model, ds_test)

if __name__ == '__main__':
    main()
