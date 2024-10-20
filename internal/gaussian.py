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
        
        self.means = nn.Parameter(self.means, requires_grad=self.require_grad)
        self.shs = nn.Parameter(self.shs, requires_grad=self.require_grad)
        self.scales = nn.Parameter(self.scales, requires_grad=self.require_grad)
        self.rotations = nn.Parameter(self.rotations, requires_grad=self.require_grad)
        self.opacities = nn.Parameter(self.opacities, requires_grad=self.require_grad)
    
    
    def reset_parameters(self) -> None:
        self.init_empty()
        
        
    def init_randomize_manual(self, means_rg=[.0, 1.0], shs_rg=[.0, 1.0], scales_rg=[.0, 1.0], rotation_rg=[.0, 1.0], opacity_rg=[.0, 1.0]):
        init.uniform(self.means, means_rg[0], means_rg[1])
        init.uniform(self.shs, shs_rg[0], shs_rg[1])
        init.uniform(self.scales, scales_rg[0], scales_rg[1])
        init.uniform(self.rotations, rotation_rg[0], rotation_rg[1])
        init.uniform(self.opacities, opacity_rg[0], opacity_rg[1])
            
    
    def init_randomize(self, means_rg=[.0, 1.0], shs_rg=[.0, 1.0], scales_rg=[.05, .5], rotation_rg=[.0, 1.0], opacity_rg=[.0, .01]):
        self.means = torch.rand([self.n, 3], device=self.device, dtype=torch.float) * (means_rg[1] - means_rg[0]) + means_rg[0]
        self.shs = torch.rand([self.n, self.max_sh_degree], device=self.device, dtype=torch.float) * (shs_rg[1] - shs_rg[0]) + shs_rg[0]
        self.scales = torch.rand([self.n, 3], device=self.device, dtype=torch.float) * (scales_rg[1] - scales_rg[0]) + scales_rg[0]
        self.rotations = torch.rand([self.n, 4], device=self.device, dtype=torch.float) * (rotation_rg[1] - rotation_rg[0]) + rotation_rg[0]
        self.opacities = torch.rand([self.n], device=self.device, dtype=torch.float) * (opacity_rg[1] - opacity_rg[0]) + opacity_rg[0]
        
        
    def init_from_file(self, data_path):
        self.means = torch.load(os.path.join(data_path, "means.pt")).to(self.device)
        self.shs = torch.load(os.path.join(data_path, "shs.pt")).to(self.device)
        self.scales = torch.load(os.path.join(data_path, "scales.pt")).to(self.device)
        self.rotations = torch.load(os.path.join(data_path, "rotations.pt")).to(self.device)
        self.opacities = torch.load(os.path.join(data_path, "opacities.pt")).to(self.device)
        self.n = self.means.shape[0]


    def init_empty(self):
        self.means = torch.zeros([self.n, 3], device=self.device, dtype=torch.float)
        self.shs = torch.zeros([self.n, self.max_sh_degree], device=self.device, dtype=torch.float)
        self.scales = torch.ones([self.n, 3], device=self.device, dtype=torch.float)
        self.rotations = torch.ones([self.n, 4], device=self.device, dtype=torch.float)
        self.opacities = torch.zeros([self.n], device=self.device, dtype=torch.float)

    
    def save_gaussians(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        torch.save(self.means, os.path.join(data_path, "means.pt"))
        torch.save(self.shs, os.path.join(data_path, "shs.pt"))
        torch.save(self.scales, os.path.join(data_path, "scales.pt"))
        torch.save(self.rotations, os.path.join(data_path, "rotations.pt"))
        torch.save(self.opacities, os.path.join(data_path, "opacities.pt"))
        
    