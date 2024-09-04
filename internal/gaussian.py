import torch
import torch.nn as nn
import torch.functional as F
from .utils import eval_gaussian_3d, eval_sh

class GaussianModel():
    def __init__(self, n : int) -> None:
        
        self.n = n
        self.max_sh_degree = 4
        means = torch.zeros((self.n, 3))
        shs = torch.zeros((self.n, (self.max_sh_degree + 1) ** 2))
        scales = torch.zeros((self.n, 3))
        rotations = torch.zeros((self.n, 4))
        opacities = torch.zeros((self.n,))
        
        # for test
        means = torch.rand((self.n, 3)) * 3 + 1
        shs = torch.rand((self.n, (self.max_sh_degree + 1) ** 2)) + 1
        scales = torch.rand((self.n, 3)) * 2 + 1
        rotations = torch.rand((self.n, 4))
        opacities = torch.ones((self.n,))
        
        # test
        # means = torch.tensor([[4., 5., 6.], [9., 3., 1.]])
        # scales = torch.tensor([[1., 1., 2.], [1., 1., 1.]])
        # rotations = torch.tensor([[0., 1., 0., 0.], [0., 1., 0., 0.]])
        # opacities = torch.tensor([1., 1.])

        self.means = nn.Parameter(means.requires_grad_(True)).to("cuda")
        self.shs = nn.Parameter(shs.requires_grad_(True)).to("cuda")
        self.scales = nn.Parameter(scales.requires_grad_(True)).to("cuda")
        self.rotations = nn.Parameter(rotations.requires_grad_(True)).to("cuda")
        self.opacities = nn.Parameter(opacities.requires_grad_(True)).to("cuda")

    
    def get_eta(self, x : torch.Tensor):
        
        # TODO: add shs here
        
        x_with_grad = x.clone().detach().requires_grad_(True)  # [M, 3]
        
        etas = eval_gaussian_3d(self.means, self.scales, self.rotations, self.opacities, self.shs, x_with_grad) 
        
        # print(f"\netas: {etas.shape}\n {etas}")
        
        d_etas = torch.autograd.grad(etas.sum(), x_with_grad)[0]
        
        # d_eta = torch.cat([torch.autograd.grad(eta, x_grad) for eta in etas], 0)  # [M, 3]
        
        return etas + 1, d_etas  # [M], [M, 3]
    
    
    def ray_trace(self, rays_o, rays_dir, d_s, N_samples):
        rays_points = []
        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        for _ in range(N_samples):
            rays_points.append(rays_o)
            etas, d_etas = self.get_eta(rays_o)
            rays_o = rays_o + rays_dir / etas * d_s
            rays_dir = rays_dir + d_etas * d_s
    
        return torch.cat(rays_points, dim=0)