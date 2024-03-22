import torch
import torch.nn as nn
import torch.optim as optim
from model.model import MILModel
from model.model import MLIMNISTModel
from data.data_utils.transformations import get_transformation

def setup_model_training(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "TCGA"]:
        model = MILModel(args.dataset, args.mil_type, args.pooling_type)
        optimizer = optim.SGD(model.parameters(), 
                              lr=args.learning_rate, 
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay)
        criterion = nn.BCELoss()
        transformation = get_transformation(device)
    elif args.dataset == "MNIST":
        model = MLIMNISTModel(args.mil_type, args.pooling_type)
        optimizer = optim.Adam(model.parameters(), 
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay, 
                               betas=(args.beta_1, args.beta_2))
        criterion = nn.BCELoss()
        transformation = get_transformation(device, normalization_params=255.)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported!")

    return model, device, optimizer, criterion, transformation


def setup_model_training_cv(dataset, dict_params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mil_type = dict_params['mil_type']
    pooling_type = dict_params['pooling_type']
    learning_rate = dict_params['learning_rate']
    momentum = dict_params['momentum']
    weight_decay = dict_params['weight_decay']
    beta_1 = dict_params.get('beta_1', None)
    beta_2 = dict_params.get('beta_2', None)
    optimizer_name = dict_params['optimizer']

    if dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "TCGA"]:
        model = MILModel(dataset, mil_type, pooling_type)
        
    elif dataset == "MNIST":
        model = MLIMNISTModel(mil_type, pooling_type)
        
    else:
        raise ValueError(f"Dataset {dataset} not supported!")

    model.to(device)

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)

    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay,
                       betas=(beta_1, beta_2))
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported!")

    criterion = nn.BCELoss()



    return model, optimizer, criterion
