import torch
import numpy as np

def get_transformation(device, normalization_params=None):

    def transformation(bag, label):

        #normalization
        if normalization_params:
            transformed_bag = []
            for instance in bag:
                transformed_bag.append(instance / normalization_params) # dimensionality needs to match
        else:
            transformed_bag = bag

        return torch.tensor(np.array(transformed_bag), dtype = torch.float32, device=device), torch.tensor(label, dtype = torch.float32, device=device)

    return transformation