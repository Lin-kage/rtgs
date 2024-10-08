import torch
from .utils import get_eta_autograd, get_eta_manual


def ray_trace(gaussians, rays_o, rays_dir, d_s, N_samples, auto_grad=False):
    
    d_s = torch.tensor(d_s).to("cuda")
    
    rays_points = []
    rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
    for _ in range(N_samples):
        rays_points.append(rays_o)
        
        if auto_grad:
            etas, d_etas = get_eta_autograd(gaussians, rays_o)
        else:
            etas, d_etas = get_eta_manual(gaussians, rays_o)
        
        rays_o = rays_o + rays_dir / etas[:, None] * d_s
        rays_dir = rays_dir + d_etas * d_s
        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)

    return torch.cat(rays_points, dim=0)