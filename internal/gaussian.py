import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
import os

class Gaussian(nn.Module):
    def __init__(self, n: int = 1, ckpt=None, device = "cuda") -> None:
        super().__init__()
        self.device = device
        self.n = n
        
        if ckpt is not None:
           self.means = ckpt['state_dict']['gaussians.means'].to(self.device)
           self.scales = self.scale_activation(ckpt['state_dict']['gaussians.scales'].to(self.device))
           self.rotations = self.rotation_activation(ckpt['state_dict']['gaussians.rotations'].to(self.device))
           self.opacities = self.opacity_activation(ckpt['state_dict']['gaussians.opacities'].to(self.device))
           self.n = self.means.shape[0]
        else:
            self.means = torch.zeros((self.n, 3), device=self.device).to(self.device)
            self.scales = torch.ones((self.n, 3), device=self.device).to(self.device) * 0.1
            self.rotations = torch.ones((self.n, 4), device=self.device).to(self.device) 
            self.opacities = torch.zeros(self.n, device=self.device).to(self.device)


    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    
    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)
    
    
    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)