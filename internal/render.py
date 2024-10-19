import torch
from .utils import get_eta_autograd, get_eta_manual


def ray_trace(rays_o : torch.Tensor, rays_dir : torch.Tensor, d_s, d_steps, lum_field_fn, eta_field_fn, auto_grad=False):
    
    rays_points = []
    rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
    for _ in range(d_steps):
        rays_points.append(rays_o)
        
        if auto_grad:
            with torch.autograd():
                rays_o_grad = rays_o.clone().requires_grad_(True)
                etas = eta_field_fn(rays_o_grad)
                d_etas = torch.autograd.grad(etas.sum(), rays_o_grad)
        else:
            etas, d_etas = eta_field_fn(rays_o)
        
        rays_o = rays_o + rays_dir / etas[:, None] * d_s
        rays_dir = rays_dir + d_etas * d_s
        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        
    rays_points = torch.cat(rays_points, dim=0)
    rays_lum = lum_field_fn(rays_points).reshape(rays_o.shape[0], -1).sum(dim=-1)
    
    return rays_lum
        
    