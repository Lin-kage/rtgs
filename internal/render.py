import torch
from .utils import get_eta_autograd, get_eta_manual
from .viewer import plot_3d


def ray_trace(rays_o : torch.Tensor, rays_dir : torch.Tensor, d_s, d_steps, lum_field_fn, eta_field_fn, auto_grad=False):
    
    rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
    rays_lum = torch.zeros_like(rays_o[:, 0])
    for _ in range(d_steps):
        
        if auto_grad:
            with torch.enable_grad():
                rays_o_grad = rays_o.clone().requires_grad_(True)
                etas = eta_field_fn(rays_o_grad)
                d_etas = torch.autograd.grad(etas.sum(), rays_o_grad)[0]
        else:
            etas, d_etas = eta_field_fn(rays_o)
        
        rays_o = rays_o + rays_dir / etas[:, None] * d_s
        rays_dir = rays_dir + d_etas * d_s
        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        rays_lum += lum_field_fn(rays_o) * d_s
        
    
    # precision = 16
    # x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
    # points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
    # eta, _ = eta_field_fn(points)
    # plot_3d(eta, precision, torch.cat(rays_points, dim=0))
    
    return rays_lum
        
    