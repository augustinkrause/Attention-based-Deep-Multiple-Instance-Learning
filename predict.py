import argparse
import torch
import math 
import os
from model.model import MILModel
from model.model import MLIMNISTModel

def confidence_measure(x):  #maps (.5,1) to (0,1)

        return 2*x-1


def predict(args):
    
    model = args.model
      
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
    parser.add_argument("--model", 
                        type = torch.Module)
    
    args = parser.parse_args()
    
    if args.input == None:
        raise ValueError("Specify an input sample.")
    if args.model == None:
        raise ValueError("Specify a model")
 
    prediction, confidence = predict(args)
    print(f"Prediction: {prediction}, Confidence: {confidence}")
    print()

if __name__ == "__main__":
    main()
