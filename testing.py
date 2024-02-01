import argparse
import torch
from training import test, evaluate
import os
from model.model import MILModel
from model.model import MLIMNISTModel
from data.data import load_data
from data.data_utils.transformations import get_transformation


def main():
    """
        This can be called by running "python -m testing [PARAMETERS]
    """
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--parameter-path', default = "model/model_parameters", type=str)
    args = parser.parse_args()
    print("Args parsed!", flush=True)
    print(f"Args: {args}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(args.parameter_path, f"{args.dataset}_{args.mil_type}_{args.pooling_type}.pt")
    load_model = torch.load(path, map_location=device)
    if args.dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "TCGA"]: 
        model = MILModel(args.dataset, args.mil_type, args.pooling_type)
    elif args.dataset == "MNIST": 
        model = MLIMNISTModel(args.mil_type, args.pooling_type)
	
    model.load_state_dict(load_model)
    model.to(device)
    model.eval()

    transformation = get_transformation(device, normalization_params=255. if args.dataset == "MNIST" else None)
    _, ds_test = load_data(args.dataset, transformation=transformation, n_train = args.n_train, n_test = args.n_test)

    y_pred, y_true, total_correct, total_samples = test(model, ds_test)
    accuracy, precision, recall, f1, roc = evaluate(y_pred, y_true, total_correct, total_samples)

	
if __name__ == '__main__':
	main()