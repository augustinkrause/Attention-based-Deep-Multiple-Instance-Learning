from training import train_single_epoch, test
from model.model_utils.setup_model_training import setup_model_training
from data.data import load_data


def train_apply(method = "attention", dataset = "MNIST", mil_type = "embeddings_based", n_train = 73, n_test=19, n_epochs=20):

	# TODO: Do we want model selection here?
	# TODO: for now I've just set the default parameters here...
	args = {
		"dataset" : dataset,
		"mil_type": mil_type,
		"pooling_type": method,
		"learning_rate": 0.0005,
		"momentum": 0.9,
		"weight_decay": 0.0001,
		"beta_1": 0.9,
		"beta_2": 0.999
	}

	# set up training and load data
	model, device, optimizer, criterion, transformation = setup_model_training(args)
	model.to(device)
	ds_train, ds_test = load_data(dataset, transformation=transformation, n_train = n_train, n_test = n_test)

	# train model
	model.train()
	for epoch in range(1, n_epochs +1):
		train_single_epoch(model, ds_train, criterion, optimizer, 100, epoch)

	# test model
	return test(model, ds_test)[0] # return only the predictions