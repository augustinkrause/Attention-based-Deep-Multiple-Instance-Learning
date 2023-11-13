import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



def get_transformation_MNIST(device):

    def transform_mnist_bags_tensor(bag, label):

        bag_l = []

        for instance in bag:
            bag_l.append(instance/255)

        return torch.tensor(np.array(bag_l), dtype = torch.float32).to(device), torch.tensor(label, dtype = torch.float32).to(device)


    return transform_mnist_bags_tensor
