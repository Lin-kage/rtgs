import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F

class Gaussian(nn.Module):
    def __init__(self, n : int, device = "cuda") -> None:
        super().__init__()
        self.device = device
        self.n = n
        self.max_sh_degree = 4
        self.means = nn.Parameter(torch.empty((n, 3), device=device, dtype=torch.float, requires_grad=True))
        self.shs = nn.Parameter(torch.empty((n, self.max_sh_degree), device=device, dtype=torch.float, requires_grad=True))
        self.scales = nn.Parameter(torch.empty((n, 3), device=device, dtype=torch.float, requires_grad=True))
        self.rotations = nn.Parameter(torch.empty((n, 4), device=device, dtype=torch.float, requires_grad=True))
        self.opacities = nn.Parameter(torch.empty((n, ), device=device, dtype=torch.float, requires_grad=True))
    
    
    def reset_parameters(self) -> None:
        self.init_randomize()
            
    
    def init_randomize(self):
        init._no_grad_uniform_(self.means, 0.0, 1.0)
        init._no_grad_uniform_(self.shs,0., 1.)
        init._no_grad_uniform_(self.scales, .05, .5)
        init._no_grad_uniform_(self.rotations, 0., 1.)
        init._no_grad_uniform_(self.opacities, .0, .1)

    
    
    