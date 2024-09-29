import torch

def print_tensor(value_name : str, x : torch.Tensor):
    print(f"{value_name}: {x.shape}\n{x.tolist()}\n")