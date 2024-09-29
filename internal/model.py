import torch
import lightning.pytorch as pl
import numpy as np  
from .gaussian import Gaussian
from .render import ray_trace
from .utils import get_eta_autograd

class GaussianModel(pl.LightningModule):    

    def __init__(self, 
                 lum_field_fn, 
                 lr = 1e-5, n_gaussians = 100, d_steps = 500):
        super().__init__()
        
        self.lr = lr
        self.n_gaussians = n_gaussians
        self.d_steps = d_steps
        self.lum_field_fn = lum_field_fn
        
        self.gaussians = Gaussian(n_gaussians)
        self.gaussians.init_randomize()
        
        self.to(self.gaussians.device)
        
        
    def forward(self, input):   
        
        d_s = 1.2 / self.d_steps
        
        rays_o = input[:, :3]
        rays_dir = input[:, 3:6]
        rays_lum = torch.zeros_like(input[:, 6:], device=rays_o.device)

        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        
        for _ in range(self.d_steps):
            
            etas, d_etas = get_eta_autograd(self.gaussians, rays_o)
            
            rays_o = rays_o + rays_dir / etas[:, None] * d_s
            rays_dir = rays_dir + d_etas * d_s
            rays_lum = rays_lum + self.lum_field_fn(rays_o) * d_s
            rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
            
        return rays_lum
    
    
    
    def training_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6:]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
            
            
    def validation_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6:]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
        
        
    def test_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6:]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('test_loss', loss, prog_bar=True)
        
        return loss
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)