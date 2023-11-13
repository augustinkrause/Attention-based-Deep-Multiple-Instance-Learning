import torch
import torch.nn as nn


class MILModel(nn.Module):
    def __init__(self, dataset = "MUSK1", mil_type  = "embedding_based", pooling_type = "attention"):
        super(MILModel, self).__init__()
        
        if dataset == "MUSK1" or dataset == "MUSK2":
            self.input_size = 166
        elif dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":
            self.input_size = 230
        else:
            raise ValueError("Dataset is not a classical MLI dataset")
        
        if mil_type  == "embedding_based":

            if pooling_type != "max" and pooling_type != "mean" and pooling_type != "attention" and pooling_type != "gated attention":

                raise ValueError(f"Pooling type {pooling_type} it not supported for MIL type {mil_type}")

        elif mil_type  == "instance_based":

            if pooling_type != "max" and pooling_type != "mean":

                raise ValueError(f"Pooling type {pooling_type} it not supported for MIL type {mil_type}")

        else:
            raise ValueError(f"MIL type {mil_type} it not supported")
        
        self.pooling_type = pooling_type
        self.mil_type = mil_type
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        #MIL-Pooling
        if pooling_type == "max":
            self.mil_layer = nn.AdaptiveMaxPool1d(1)

        elif pooling_type == "mean":
            self.mil_layer = nn.AdaptiveAvgPool1d(1)

        elif pooling_type == "attention":
            self.mil_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
            )
        
        elif pooling_type == "gated attention":
            self.V = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            )
            self.U = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            )
            self.mil_layer = nn.Linear(64,1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid())


    def forward(self, x):
        x = self.feature_extractor(torch.tensor(x))

        if self.mil_type == "embedding_based":
            if self.pooling_type == "mean" or self.pooling_type == "max":
                x = self.mil_layer(x.T).flatten()

            elif self.pooling_type == "attention":
                a = self.mil_layer(x)
                x = (a * x).sum(dim = 0)

            elif self.pooling_type == "gated attention":
                U = self.U(x)
                V = self.V(x)
                a = nn.functional.softmax(self.mil_layer(U*V), dim = 0)
                x = (a * x).sum(dim = 0)

            x = self.classifier(x) 
        
        elif self.mil_type  == "instance_based":
            x = self.classifier(x)
            x = self.mil_layer(x.T).flatten()

        return x
