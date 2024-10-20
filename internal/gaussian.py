import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
import os

class Gaussian(nn.Module):
    def __init__(self, n: int = 1, device = "cuda", init_from_file=None, init_random=False, require_grad=True) -> None:
        super().__init__()
        self.device = device
        self.n = n
        self.max_sh_degree = 4
        self.require_grad = require_grad
        
        if init_from_file is not None:
            self.init_from_file(init_from_file)
        elif init_random:
            self.init_randomize()
        else:
            self.init_empty()
        
        self.means = nn.Parameter(self.means, device=self.device, dtype=torch.float, requires_grad=self.require_grad))
        self.shs = nn.Parameter(self.shs, device=self.device, dtype=torch.float, requires_grad=self.require_grad))
        self.scales = nn.Parameter(self.scales, device=self.device, dtype=torch.float, requires_grad=self.require_grad))
        self.rotations = nn.Parameter(self.rotations, device=self.device, dtype=torch.float, requires_grad=self.require_grad))
        self.opacities = nn.Parameter(self.opacities, device=self.device, dtype=torch.float, requires_grad=self.require_grad))
    
    
    def reset_parameters(self) -> None:
        self.init_empty()
            
    
    def init_randomize(self, means_rg=[.0, 1.0], shs_rg=[.0, 1.0], scales_rg=[.0, 1.0], rotation_rg=[.0, 1.0], opacity_rg=[.0, 1]):
        self.means = torch.rand([self.n, 3]) * (means_rg[1] - means_rg[0]) + means_rg[0]
        self.shs = torch.rand([self.n, self.max_sh_degree]) * (shs_rg[1] - shs_rg[0]) + shs_rg[0]
        self.scales = torch.rand([self.n, 3]) * (scales_rg[1] - scales_rg[0]) + scales_rg[0]
        self.rotations = torch.rand([self.n, 4]) * (rotation_rg[1] - rotation_rg[0]) + rotation_rg[0]
        self.opacities([self.n]) * (opacity_rg[1] - opacity_rg[0]) + opacity_rg[0]
        
        
    def init_from_file(self, data_path):
        self.means = torch.load(os.path.join(data_path, "means.pt"))
        self.shs = torch.load(os.path.join(data_path, "shs.pt"))
        self.scales = torch.load(os.path.join(data_path, "scales.pt"))
        self.rotations = torch.load(os.path.join(data_path, "rotations.pt"))
        self.opacities = torch.load(os.path.join(data_path, "opacities.pt"))
        self.n = self.means.shape[0]


    def init_empty(self):
        self.means = torch.zeros([self.n, 3])
        self.shs = torch.zeros([self.n, self.max_sh_degree])
        self.scales = torch.zeros([self.n, 3])
        self.rotations = torch.zeros([self.n, 4])
        self.opacities = torch.zeros([self.n])

    
    def save_gaussians(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        torch.save(self.means, os.path.join(data_path, "means.pt"))
        torch.save(self.shs, os.path.join(data_path, "shs.pt"))
        torch.save(self.scales, os.path.join(data_path, "scales.pt"))
        torch.save(self.rotations, os.path.join(data_path, "rotations.pt"))
        torch.save(self.opacities, os.path.join(data_path, "opacities.pt"))
        
    