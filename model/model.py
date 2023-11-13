import torch
import torch.nn as nn
import numpy as np


class MLIMNISTModel(nn.Module):
    def __init__(self, mil_type  = "embedding_based", pooling_type = "attention"):
        super(MLIMNISTModel, self).__init__()

        if mil_type  == "embedding_based":

            if pooling_type != "max" and pooling_type != "mean" and pooling_type != "attention":

                raise ValueError(f'Pooling type {pooling_type} it not supported for MIL type {mil_type}')

        elif mil_type  == "instance_based":

            if pooling_type != "max" and pooling_type != "mean":

                raise ValueError(f'Pooling type {pooling_type} it not supported for MIL type {mil_type}')

        else:
            raise ValueError(f'MIL type {mil_type} it not supported')


        self.mil_type = mil_type
        self.pooling_type = pooling_type
        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()

        # paramaterized layers
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.conv2d2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 1)

        if self.pooling_type  == "attention":

            # paramaterized layers for attention
            self.attention_V = nn.Linear(500, 128)
            self.attention_U = nn.Linear(500, 128)
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


       

        return x

