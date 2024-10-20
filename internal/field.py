import torch
from scipy import ndimage
from .gaussian import Gaussian
from .utils import get_eta_manual, eval_gaussian_3d


class Grid3D():
    """
        An interpolate grid
        the input grid_val must be N*N*N
    """
    
    def __init__(self, grid_val, cval=0.):
        self.grid_val = grid_val
        shape = grid_val.shape
        
        if shape[0] != shape[1] or shape[1] != shape[2] or len(shape) != 3:
            raise ValueError("The input grid_val must be N*N*N")
        
        self.grid_res = shape[0]
        self.cval = cval
        
        
    def interp(self, x, order=1):
        coords = x * (self.grid_res-1)
        out = ndimage.map_coordinates(self.grid_val, coords.T.cpu(), order=order, cval=self.cval, )
        return torch.tensor(out, device=x.device)
    
    
class TensorGrid3D():
    """
        An interpolate tensor gird
        the input grid_val must be N*N*N
        can be used for autograd
    """
    
    def __init__(self, grid_val: torch.Tensor, cval=0.):
        self.grid_val = grid_val
        shape = grid_val.shape
        
        if len(shape) != 3 or shape[0] != shape[1] or shape[1] != shape[2]:
            raise ValueError("The input grid_val must be N*N*N")
        
        self.grid_res = shape[0]
        self.cval = cval
        
    
    def interp_linear(self, x : torch.Tensor):
        
        self.grid_val = self.grid_val.to(x.device)
        coords = x * (self.grid_res-1)
        coords_int = torch.floor(coords).to(x.device)
        coords_out_of_range = torch.any(coords_int < 0, dim=-1) | torch.any(coords_int >= self.grid_res - 1, dim=-1)
        coords_in_range = ~ coords_out_of_range
         
        result = torch.zeros([x.shape[0]], device=x.device)
        result[coords_out_of_range] = self.cval
        
        coords = coords[coords_in_range]
        coords_int = coords_int[coords_in_range].long()
        c_d = (coords - coords_int)
        x_d, y_d, z_d = c_d[:, 0], c_d[:, 1], c_d[:, 2]
        
        c000 = self.grid_val[coords_int[:,0], coords_int[:,1], coords_int[:, 2]]
        c100 = self.grid_val[coords_int[:,0]+1, coords_int[:,1], coords_int[:, 2]]
        c010 = self.grid_val[coords_int[:,0], coords_int[:,1]+1, coords_int[:, 2]]
        c001 = self.grid_val[coords_int[:,0], coords_int[:,1], coords_int[:, 2]+1]
        c110 = self.grid_val[coords_int[:,0]+1, coords_int[:,1]+1, coords_int[:, 2]]
        c011 = self.grid_val[coords_int[:,0], coords_int[:,1]+1, coords_int[:, 2]+1]
        c101 = self.grid_val[coords_int[:,0]+1, coords_int[:,1], coords_int[:, 2]+1]
        c111 = self.grid_val[coords_int[:,0]+1, coords_int[:,1]+1, coords_int[:, 2]+1]
        
        result[coords_in_range] = \
            c000 * (1-x_d) * (1-y_d) * (1-z_d) + \
            c100 * x_d * (1-y_d) * (1-z_d) + \
            c010 * (1-x_d) * y_d * (1-z_d) + \
            c001 * (1-x_d) * (1-y_d) * z_d + \
            c110 * x_d * y_d * (1-z_d) + \
            c011 * (1-x_d) * y_d * z_d + \
            c101 * x_d * (1-y_d) * z_d + \
            c111 * x_d * y_d * z_d
            
        return result
                
            
class FieldGenerator():
    """
        using 3dgs to generate a field
    """
    
    def __init__(self, n_gaussian=1, device = "cuda", init_from_file=None, init_random=True):
        
        self.gaussians = Gaussian(n_gaussian, device, init_from_file, init_random, False)
        
        
    def get_lum(self, points):
        
        lum_field = eval_gaussian_3d(self.gaussians, points)
        
        return lum_field
        
        
    def get_eta(self, points):
        
        eta, d_eta = get_eta_manual(self.gaussians, points)
        
        return eta, d_eta