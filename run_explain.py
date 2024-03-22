import itertools
import argparse
import torch
import sys
import os
from data.data import load_data

from model.model import MILModel
from model.model import MLIMNISTModel
from data.data_utils.transformations import get_transformation
from explain import explain


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_folder = "data/explains"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
   

    for dataset, mil_type, pooling_type in itertools.product(
                ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "MNIST", "TCGA"],
                ["embedding_based", "instance_based"],
                ["max", "mean", "attention", "gated_attention"]
        ):
        if mil_type == "instance_based" and pooling_type in ["attention", "gated_attention"]:
            continue
        path = os.path.join(output_folder, f"explain_{dataset}_{mil_type}_{pooling_type}.txt")

        with open(path, "w") as f:
            sys.stdout = f   
            
            if dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "TCGA"]:
                model = MILModel(dataset, mil_type, pooling_type, return_attentions = True)
                transformation = get_transformation(device)
            elif dataset == "MNIST":
                model = MLIMNISTModel(mil_type, pooling_type, return_attentions = True)
                transformation = get_transformation(device, normalization_params=255.)

            if dataset == "TCGA":
                model_path = os.path.join("model/model_parameters",f"{dataset}_{mil_type}_{pooling_type}-seed1.pt")
            else:
                model_path = os.path.join("model/model_parameters",f"{dataset}_{mil_type}_{pooling_type}.pt")

            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            print(f"Dataset: {dataset}, MIL Type: {mil_type}, Pooling Type: {pooling_type}")
            _, ds_eval = load_data(dataset, transformation=transformation)

            plot = True if dataset == "MNIST" else False
            output_plot_folder = os.path.join(os.getcwd(), "data", "plots")
            if not os.path.exists(output_plot_folder):
                os.makedirs(output_plot_folder)
            
            for idx, (bag, label) in enumerate(ds_eval):
                filename= f'{dataset}_{idx}_{dataset}_{mil_type}_{pooling_type}_analysis_plot'

                if idx >= 20:
                    plot = False
                sensitivty_arr= explain(bag, label, model, plot, dataset, pooling_type, output_plot_folder, filename)

                print()
                print("Bag number: ", idx)
                print("Sensitivty mean: ", sensitivty_arr.mean())
                print("Sensitivty standard deviation: ", sensitivty_arr.std())
                print()

            print()


        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()