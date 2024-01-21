import torch
import torch.nn as nn


class MILModel(nn.Module):
    def __init__(self, dataset = "MUSK1", mil_type  = "embedding_based", pooling_type = "attention", return_attentions = False):
        super(MILModel, self).__init__()
        
        if dataset == "MUSK1" or dataset == "MUSK2":
            self.input_size = 166
        elif dataset == "FOX" or dataset == "ELEPHANT" or dataset == "TIGER":
            self.input_size = 230
        else:
            raise ValueError("Dataset is not a classical MLI dataset")
        
        if mil_type  == "embedding_based":

            if pooling_type != "max" and pooling_type != "mean" and pooling_type != "attention" and pooling_type != "gated_attention":

                raise ValueError(f"Pooling type {pooling_type} is not supported for MIL type {mil_type}")

        elif mil_type  == "instance_based":

            if pooling_type != "max" and pooling_type != "mean":

                raise ValueError(f"Pooling type {pooling_type} it not supported for MIL type {mil_type}")

        else:
            raise ValueError(f"MIL type {mil_type} it not supported")
        
        self.pooling_type = pooling_type
        self.mil_type = mil_type
        self.return_attentions = return_attentions
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
            nn.Softmax(dim=0)
            )
        
        elif pooling_type == "gated_attention":
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
        
        x = self.feature_extractor(x)

        if self.mil_type == "embedding_based":
            if self.pooling_type == "mean" or self.pooling_type == "max":
                x = self.mil_layer(x.T).flatten()

            elif self.pooling_type == "attention":
                a = self.mil_layer(x)
                x = (a * x).sum(dim = 0)

            elif self.pooling_type == "gated_attention":
                U = self.U(x)
                V = self.V(x)
                a = nn.functional.softmax(self.mil_layer(U*V), dim = 0)
                x = (a * x).sum(dim = 0)

            x = self.classifier(x) 
        
        elif self.mil_type  == "instance_based":
            x = self.classifier(x)
            x = self.mil_layer(x.T).flatten()

        if self.return_attentions and (self.pooling_type == "attention" or self.pooling_type == "gated_attention"):
            return x, a
        else:
            return x
    

class MLIMNISTModel(nn.Module):
    def __init__(self, mil_type  = "embedding_based", pooling_type = "gated_attention", return_attentions = False):
        super(MLIMNISTModel, self).__init__()

        if mil_type  == "embedding_based":

            if pooling_type != "max" and pooling_type != "mean" and pooling_type != "gated_attention" and pooling_type != "attention":

                raise ValueError(f'Pooling type {pooling_type} it not supported for MIL type {mil_type}')

        elif mil_type  == "instance_based":

            if pooling_type != "max" and pooling_type != "mean":

                raise ValueError(f'Pooling type {pooling_type} it not supported for MIL type {mil_type}')

        else:
            raise ValueError(f'MIL type {mil_type} it not supported')

        self.mil_type = mil_type
        self.pooling_type = pooling_type
        self.return_attentions = return_attentions
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()

        # paramaterized layers
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.conv2d2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 1)

        if self.pooling_type  == "gated_attention":

            # paramaterized layers for attention
            self.attention_V = nn.Linear(500, 128)
            self.attention_U = nn.Linear(500, 128)
            self.attention_w = nn.Linear(128, 1)

        elif self.pooling_type == "attention":
            self.attention_V = nn.Linear(500, 128)
            self.attention_w = nn.Linear(128, 1)

    def forward(self, x):

        x= self.conv2d1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2d2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        if self.mil_type  == "embedding_based":

            if self.pooling_type == "mean":
                x = torch.mean(x, dim = 0)
            elif self.pooling_type == "max":
                x, _ = torch.max(x, dim = 0)
            elif self.pooling_type == "attention":
                a_1 = self.attention_V(x)
                a_1 = self.tanh(a_1)
                a = self.attention_w(a_1)
                a = self.softmax(a)
                print()
                print(a)
                x = torch.sum(a * x, dim = 0)
            elif self.pooling_type == "gated_attention":
                a_1 = self.attention_V(x)
                a_1 = self.tanh(a_1)
                a_2 = self.attention_U(x)
                a_2 = self.sigm(a_2)
                a = a_1 * a_2
                a = self.attention_w(a)
                a = self.softmax(a) 
                x = a * x
                x = torch.sum(x, dim = 0)
                
            x = self.fc2(x)
            x = self.sigm(x)

        elif self.mil_type  == "instance_based":
            
            x = self.fc2(x)
            x = self.sigm(x)
            
            if self.pooling_type == "mean":
                x = torch.mean(x, dim = 0)

            elif self.pooling_type == "max":
                x, _ = torch.max(x, dim = 0)

        if self.return_attentions and (self.pooling_type == "attention" or self.pooling_type == "gated_attention"):
            return x, a
        else:
            return x
