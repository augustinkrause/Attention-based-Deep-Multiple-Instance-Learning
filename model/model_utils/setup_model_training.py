import torch
import torch.nn as nn
import torch.optim as optim
from model.model import MILModel
from model.model import MLIMNISTModel
from data.data_utils.transformations import get_transformation

def setup_model_training(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX"]:
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
        