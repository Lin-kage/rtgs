import torch
import lightning.pytorch as pl
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F 
from .gaussian import Gaussian
from .render import ray_trace
from .utils import get_eta_autograd, get_eta_manual
from .viewer import plot_3d
from .field import FieldGenerator, TensorGrid3D

class GaussianModel(pl.LightningModule):    

    def __init__(self, 
                 lum_field_fn, 
                 lr = 1e-5, d_steps = 500, 
                 view_per_epochs = 10,
                 n_gaussians = 1,
                 device = "cuda",
                 init_randomlize = False,
                 gaussian_file = None
                 ):
        super().__init__()
        
        self.lr = lr
        self.n_gaussians = n_gaussians
        self.d_steps = d_steps
        self.lum_field_fn = lum_field_fn
        self.d_s = 1.2 / self.d_steps
        self.ave_train_loss_cnt = 0
        self.ave_train_loss = 0.
        
        if gaussian_file is not None:
            self.gaussians = Gaussian(n=n_gaussians, device=device, init_from_file=gaussian_file, require_grad=True)
        if init_randomlize:
            self.gaussians = Gaussian(n=n_gaussians, device=device, init_random=True, require_grad=True)
            self.gaussians.init_randomize_manual(scales_rg=[0.05, .5], opacity_rg=[0., 0.001])
        else:
            self.gaussians = Gaussian(n=n_gaussians, device=device, init_random=False)
        
        self.view_cnt = 0
        self.view_per_epochs = view_per_epochs
        
        self.to(self.gaussians.device)
        
        
    def forward(self, input):   
        
        rays_o = input[:, :3]
        rays_dir = input[:, 3:6]

        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        
        rays_lum = ray_trace(rays_o, rays_dir, self.d_s, self.d_steps, 
                             self.lum_field_fn, lambda x :get_eta_manual(self.gaussians, x), auto_grad=False)
            
        return rays_lum
    
    
    # ray trace step by step
    def ray_trace_lum(self, rays_o, rays_dir):
        
        rays_lum = torch.zeros_like(rays_o[:, 0], device=rays_o.device)
        
        for _ in range(self.d_steps):
            
            # etas, d_etas = get_eta_autograd(self.gaussians, rays_o)
            etas, d_etas = get_eta_manual(self.gaussians, rays_o)
            
            # with torch.no_grad():
            lums = self.lum_field_fn(rays_o)
            # lums = torch.zeros_like(rays_lum)
            rays_o = rays_o + rays_dir / etas[..., None] * self.d_s
            rays_dir = rays_dir + d_etas * self.d_s
            rays_lum = rays_lum + lums * self.d_s
            rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
            
        return rays_lum
    
    
    def training_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('train_loss', loss, prog_bar=True)
        
        self.ave_train_loss += loss
        self.ave_train_loss_cnt += 1
        
        return loss
            
            
    def validation_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
        
        
    def test_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('test_loss', loss, prog_bar=True)
        
        return loss
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
    def view_eta_field(self, precision=32):
        x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3).to(self.gaussians.device)
        
        with torch.no_grad():
            eta, _ = get_eta_manual(self.gaussians, points)
        plot_3d(eta, precision)
    
    
    def on_train_start(self):
        self.view_eta_field()
    
    
    def on_train_epoch_end(self):
        
        self.log("ave_train_loss", self.ave_train_loss/self.ave_train_loss_cnt, prog_bar=True, sync_dist=True)
        self.ave_train_loss_cnt = 0
        self.ave_train_loss = 0.
        
        self.view_cnt += 1
        if self.view_cnt % self.view_per_epochs == 0:
            self.view_eta_field()
            print("\n")
            
            
        