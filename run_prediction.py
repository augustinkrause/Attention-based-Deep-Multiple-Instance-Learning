from predict import predict
import itertools
import argparse
import torch
import sys
import os
from data.data import load_data

from model.model import MILModel
from model.model import MLIMNISTModel
from data.data_utils.transformations import get_transformation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    
    for threshold in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:

        output_folder = "data/predictions"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        path = os.path.join(output_folder, f"predictions_{threshold}.txt")

        with open(path, "w") as f:
            sys.stdout = f

            for dataset, mil_type, pooling_type in itertools.product(
                    ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "MNIST", "TCGA"],
                    ["embedding_based", "instance_based"],
                    ["max", "mean", "attention", "gated_attention"]
            ):

                if mil_type == "instance_based" and pooling_type in ["attention", "gated_attention"]:
                    continue
                
                if dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "TCGA"]:
                    model = MILModel(dataset, mil_type, pooling_type)
                    transformation = get_transformation(device)
                elif dataset == "MNIST":
                    model = MLIMNISTModel(mil_type, pooling_type,)
                    transformation = get_transformation(device, normalization_params=255.)

                if dataset == "TCGA":
                    model_path = os.path.join("model/model_parameters",f"{dataset}_{mil_type}_{pooling_type}-seed1.pt")
                else:
                    model_path = os.path.join("model/model_parameters",f"{dataset}_{mil_type}_{pooling_type}.pt")

                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()
                args.model = model

                print(f"Dataset: {dataset}, MIL Type: {mil_type}, Pooling Type: {pooling_type}")
                _, ds_eval = load_data(dataset, transformation=transformation)
                n_errors = 0
                FN = []
                FP = []
                TN = []
                TP = []
                PCT = []
                PCF = []
                NCT = []
                NCF = []
                
                total = 0
                counter = 0

                for bag, label in ds_eval:
                    args.input = bag
                    prediction, measure = predict(args)
                    total += measure
                    counter += 1

                    if prediction != label:
                        if prediction == 0:
                            FN.append(measure)
                            #print(f"False Negative, Confidence: {measure}")
                        else:
                            FP.append(measure)
                            #print(f"False Positive, Confidence: {measure}")
                        n_errors +=1
                    else:
                        if prediction == 0:
                            TN.append(measure)
                            #print(f"True Negative, Confidence: {measure}")
                        else:
                            TP.append(measure)
                            #print(f"True Positive, Confidence: {measure}")    

                    if measure > threshold:
                        if prediction == label:
                            PCT.append(measure)
                        else:
                            PCF.append(measure)
                    else:
                        if prediction == label:
                            NCT.append(measure)
                        else:
                            NCF.append(measure)
                            


                print(f"Average Confidence: {round(total/counter, ndigits=2)}")

                try:
                    print(f"False Negative average Confidence: {round(sum(FN)/len(FN), ndigits=2)}")
                except:
                    print("No False Negatives")
                
                try:
                    print(f"False Positive average Confidence: {round(sum(FP)/len(FP),ndigits=2)}")
                except:
                    print("No False Positives")

                try:
                    print(f"True Negative average Confidence: {round(sum(TN)/len(TN),ndigits=2)}")
                except:
                    print("No True Negatives")
                
                try:
                    print(f"True Positives average Confidence: {round(sum(TP)/len(TP),ndigits=2)}")
                except:
                    print("No True Positives")     
                print()
                print("Confidence predictions")

                try:
                    print("Confidence Ratio", (len(PCT) + len(PCF))/len(ds_eval))
                except:
                    print("No Data")

                try:
                    print("Positive Confidence Accuracy", len(PCT)/(len(PCT) + len(PCF)))
                except:
                    print("No Positive Confidence")

                try:
                    print("Negative Confidence Accuracy", len(NCF)/(len(NCF) + len(NCT)))
                except:
                    print("No Negative Confidence")
                
                
                
                print()
                print()
          



        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()