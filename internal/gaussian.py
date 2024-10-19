import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
import os

class Gaussian(nn.Module):
    def __init__(self, n : int, device = "cuda", require_grad=True) -> None:
        super().__init__()
        self.device = device
        self.n = n
        self.max_sh_degree = 4
        self.means = nn.Parameter(torch.empty((n, 3), device=device, dtype=torch.float, requires_grad=require_grad))
        self.shs = nn.Parameter(torch.empty((n, self.max_sh_degree), device=device, dtype=torch.float, requires_grad=require_grad))
        self.scales = nn.Parameter(torch.empty((n, 3), device=device, dtype=torch.float, requires_grad=require_grad))
        self.rotations = nn.Parameter(torch.empty((n, 4), device=device, dtype=torch.float, requires_grad=require_grad))
        self.opacities = nn.Parameter(torch.empty((n, ), device=device, dtype=torch.float, requires_grad=require_grad))
    
    
    def reset_parameters(self) -> None:
        self.init_randomize()
            
    
    def init_randomize(self, means_rg=[.0, 1.0], shs_rg=[.0, 1.0], scales_rg=[.0, 1.0], rotation_rg=[.0, 1.0], opacity_rg=[.0, 1]):
        init.uniform(self.means, means_rg[0], means_rg[1])
        init.uniform(self.shs, shs_rg[0], shs_rg[1])
        init.uniform(self.scales, scales_rg[0], scales_rg[1])
        init.uniform(self.rotations, rotation_rg[0], rotation_rg[1])
        init.uniform(self.opacities, opacity_rg[0], opacity_rg[1])
        
        
    def init_from_file(self, data_path):
        means = torch.load(os.path.join(data_path, "means.pt"))
        shs = torch.load(os.path.join(data_path, "shs.pt"))
        scales = torch.load(os.path.join(data_path, "scales.pt"))
        rotations = torch.load(os.path.join(data_path, "rotations.pt"))
        opacities = torch.load(os.path.join(data_path, "opacities.pt"))
        init.constant(self.means, means)
        init.constant(self.shs, shs)
        init.constant(self.scales, scales)
        init.constant(self.rotations, rotations)
        init.constant(self.opacities, opacities)

    
    def save_gaussians(self, data_path):
        torch.save(self.means, os.path.join(data_path, "means.pt"))
        torch.save(self.shs, os.path.join(data_path, "shs.pt"))
        torch.save(self.scales, os.path.join(data_path, "scales.pt"))
        torch.save(self.rotations, os.path.join(data_path, "rotations.pt"))
        torch.save(self.opacities, os.path.join(data_path, "opacities.pt"))
        
    