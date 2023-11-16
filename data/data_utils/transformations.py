import torch

def get_transformation(device, normalization_params=None):

    def transformation(bag, label):

        #normalization
        if normalization_params:
            transformed_bag = []
            for instance in bag:
                transformed_bag.append(instance / normalization_params) # dimensionality needs to match
        else:
            transformed_bag = bag

        return torch.tensor(transformed_bag, dtype = torch.float64, device=device), torch.tensor(label, dtype = torch.float64, device=device)

    return transformation