import torch
import numpy as np
from .viewer import plot_3d
from .field import TensorGrid3D

def print_tensor(value_name : str, x : torch.Tensor):
    print(f"{value_name}: {x.shape}\n{x.tolist()}\n")
    
    
    
def tensor_grid_test():
    
    lum_field = (np.load('./data/matern_s8/lum_field.npy', allow_pickle=False))
    
    # plot_3d(torch.Tensor(lum_field), 64, torch.zeros([1,3]))
    
    grid = TensorGrid3D(torch.from_numpy(lum_field).reshape(64,64,64))
    
    x, y, z = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing='xy')
    points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
    
    interp_lum_field = grid.interp_linear(points)
    
    plot_3d(interp_lum_field, 64, torch.zeros([1,3]))