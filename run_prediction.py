from predict import predict
import itertools
import argparse
import torch
import sys
import os
from data.data import load_data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type = torch.Tensor)
    parser.add_argument("--dataset", 
                        choices=["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "MNIST"])
    parser.add_argument("--mil-type", 
                        choices=["embedding_based", "instance_based"])
    parser.add_argument("--pooling-type", 
                        choices=["max", "mean", "attention", "gated_attention"])
    parser.add_argument("--sigma",
                        choices=[.5,1,5,10,20])   
    
    args = parser.parse_args()
    
    for sigma in ["linear", 5, 10, 15 ,20]:

        output_folder = "data/predictions"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        path = os.path.join(output_folder, f"predictions{sigma}.txt")

        with open(path, "w") as f:
            sys.stdout = f

            for dataset, mil_type, pooling_type in itertools.product(
                    ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX", "MNIST"],
                    ["embedding_based", "instance_based"],
                    ["max", "mean", "attention", "gated_attention"]
            ):
                args.dataset = dataset
                args.mil_type = mil_type
                args.pooling_type = pooling_type
                args.sigma = sigma

                if mil_type == "instance_based" and pooling_type in ["attention", "gated_attention"]:
                    continue



                print(f"Dataset: {dataset}, MIL Type: {mil_type}, Pooling Type: {pooling_type}")
                _, ds_eval = load_data(args.dataset)
                n_errors = 0
                FN = []
                FP = []
                TN = []
                TP = []
                total = 0
                counter = 0
                for bag, label in ds_eval:
                    args.input = torch.from_numpy(bag).to(dtype=torch.float32)
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

        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()