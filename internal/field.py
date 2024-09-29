import torch
from scipy import ndimage


class Grid3D():
    """
        An interpolate grid
        the input grid_val must be N*N*N
    """
    
    def __init__(self, grid_val : torch.Tensor, cval=0.):
        self.grid_val = grid_val
        shape = grid_val.shape
        
        if shape[0] != shape[1] or shape[1] != shape[2] or len(shape) != 3:
            raise ValueError("The input grid_val must be N*N*N")
        
        self.grid_res = shape[0]
        self.cval = cval
        
        
    def interp(self, x, order=1):
        coords = x * (self.grid_res-1)
        out = ndimage.map_coordinates(self.grid_val, coords.T, order=order, cval=self.cval)
        return out