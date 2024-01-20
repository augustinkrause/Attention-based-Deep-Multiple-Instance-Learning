import argparse
import torch
import math 
import os
from model.model import MILModel
from model.model import MLIMNISTModel

def confidence_measure(x):  #maps (.5,1) to (0,1)
    sigma = 10              #control how much you penalize uncertainty
    return 1/(1+math.exp(-sigma*(x-.75)))


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join("model/model_parameters",f"{args.dataset}_{args.mil_type}_{args.pooling_type}.pt")
    load_model = torch.load(path, map_location=device)

    if args.dataset in ["MUSK1", "MUSK2", "ELEPHANT", "TIGER", "FOX"]:
        model = MILModel(args.dataset, args.mil_type, args.pooling_type)

    elif args.dataset == "MNIST":
        model = MLIMNISTModel(args.mil_type, args.pooling_type)

    model.load_state_dict(load_model)
    model.eval()
    
    output = float(model(args.input)) #you can interpret this output as the probability that the bag is positive
    prediction = round(output)

    if prediction == 0:
        confidence = 1-output
    else:
        confidence = output

    return prediction, round(confidence_measure(confidence), ndigits=2)

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
    
    args = parser.parse_args()
    prediction, confidence = predict(args)
    print(f"Prediction: {prediction}, Confidence: {confidence}")

if __name__ == "__main__":
    main()
