import torch
import torch.nn as nn
import torch.functional as F

class Gaussian():
    def __init__(self, n : int, device = "cuda") -> None:
        self.device = device
        self.n = n
        self.max_sh_degree = 4
        self.means = torch.zeros((self.n, 3))
        self.shs = torch.zeros((self.n, (self.max_sh_degree + 1) ** 2))
        self.scales = torch.zeros((self.n, 3))
        self.rotations = torch.zeros((self.n, 4))
        self.opacities = torch.zeros((self.n,))
        
    
    def init_randomize(self):
        means = torch.rand((self.n, 3)) * .8 + .1
        shs = torch.rand((self.n, (self.max_sh_degree + 1) ** 2))
        scales = torch.rand((self.n, 3)) + .05
        rotations = torch.rand((self.n, 4))
        opacities = torch.ones((self.n,)) / 2.

        self.means = nn.Parameter(means.requires_grad_(True)).to(self.device)
        self.shs = nn.Parameter(shs.requires_grad_(True)).to(self.device)
        self.scales = nn.Parameter(scales.requires_grad_(True)).to(self.device)
        self.rotations = nn.Parameter(rotations.requires_grad_(True)).to(self.device)
        self.opacities = nn.Parameter(opacities.requires_grad_(True)).to(self.device)

    
    
    