from typing import List, Optional, Union
import torch
import lightning.pytorch as pl
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
import os 
from .utils import get_eta_manual
from .viewer import plot_3d
from .field import FieldGenerator, TensorGrid3D
from .config import RenderConfig, ViewConfig, OptimizationConfig, RandomizationConfig, FileConfig


class GaussianModel(pl.LightningModule):    

    def __init__(self,
                 RenderConfig: RenderConfig=RenderConfig(),
                 OptimizationConfig: OptimizationConfig=OptimizationConfig(),
                 ViewConfig: ViewConfig=ViewConfig(),
                 # for initialization
                 FileConfig: FileConfig=None,
                 RandomizationConfig: RandomizationConfig=None,
                 device = "cuda",
                 ):
        super().__init__()
        
        self.gaussians = nn.ParameterDict()
        
        self.names = ["means", "scales", "rotations", "opacities"]
        
        self.optimization_config = OptimizationConfig
        
        self.lum_field_fn = RenderConfig.lum_field_fn
        self.d_steps = RenderConfig.d_steps
        self.d_s = RenderConfig.d_s
        
        self.view_per_epoch = ViewConfig.view_per_epoch
        self.save_per_epoch = ViewConfig.save_per_epoch
        self.save_path = ViewConfig.save_path
        
        self.file_config = FileConfig
        self.randomization_config = RandomizationConfig
    
        self.automatic_optimization = False
        
        self.global_epoch = 0
        self.ave_train_loss_cnt = 0
        self.ave_train_loss = 0
        self.ave_val_loss_cnt = 0
        self.ave_val_loss = 0
        
        # self.lr = lr
        # self.n_gaussians = n_gaussians
        # self.d_steps = d_steps
        # self.lum_field_fn = lum_field_fn
        # self.d_s = 1.2 / self.d_steps
        # self.ave_train_loss_cnt = 0
        # self.ave_train_loss = 0.
        
        # if gaussian_file is not None:
        #     self.gaussians = Gaussian(n=n_gaussians, device=device, init_from_file=gaussian_file, require_grad=True)
        #     self.n_gaussians = self.gaussians.n
        # elif init_randomlize:
        #     self.gaussians = Gaussian(n=n_gaussians, device=device, init_random=False, require_grad=True)
        #     self.gaussians.init_randomize_manual(scales_rg=[0.05, .5], opacity_rg=[0., 0.001])
        # else:
        #     self.gaussians = Gaussian(n=n_gaussians, device=device, init_random=False)
        
        # self.global_epoch = 0
        # self.view_per_epochs = view_per_epochs
        
        
    def setup(self, stage: str):
        
        if self.file_config is not None:
            self.setup_from_file(self.file_config)
        elif self.randomization_config is not None:
            self.setup_randomize(self.randomization_config)
        else:
            self.setup_empty(100)
            
        
    # for configure_optimizers
    def setup_trainning(self) -> tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        
        optimization_config = self.optimization_config
        
        l = [
            {'params': [self.gaussians["means"]], 'lr': optimization_config.means_lr, "name": "means"},
            {'params': [self.gaussians["opacities"]], 'lr': optimization_config.opacities_lr, "name": "opacities"},
            {'params': [self.gaussians["scales"]], 'lr': optimization_config.scales_lr, "name": "scales"},
            {'params': [self.gaussians["rotations"]], 'lr': optimization_config.rotations_lr, "name": "rotations"},
        ]
        constant_lr_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        return [constant_lr_optimizer], []
        
        
    def forward(self, input):   
        
        rays_o = input[:, :3]
        rays_dir = input[:, 3:6]

        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        
        rays_lum = self.ray_trace_lum(rays_o, rays_dir)
            
        return rays_lum
    
    
    # ray tracing using Monte Carlo
    def ray_trace_lum(self, rays_o, rays_dir):
        
        rays_lum = torch.zeros_like(rays_o[:, 0], device=rays_o.device)
        rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
        for _ in range(self.d_steps):
            etas, d_etas = get_eta_manual(self, rays_o)
            lums = self.lum_field_fn(rays_o)
            rays_o = rays_o + rays_dir / etas[..., None] * self.d_s
            rays_dir = rays_dir + d_etas * self.d_s
            rays_lum = rays_lum + lums * self.d_s
            rays_dir = rays_dir / rays_dir.norm(dim=-1, keepdim=True)
            
        return rays_lum
    
    
    def training_step(self, batch, batch_idx):
        
        # prework
        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        lr_schedulers = self.lr_schedulers()
        
        # forward
        rays_data = batch
        rays_lum_target = rays_data[:, 6]
        rays_lum = self.forward(rays_data)
        
        # metrics
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('train_loss', loss, prog_bar=True)
        
        self.ave_train_loss += loss
        self.ave_train_loss_cnt += 1
        
        # backward
        self.manual_backward(loss)
        
        for optimizer in optimizers:
            optimizer.step()
            
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()
            
            
    def validation_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        
        self.ave_val_loss += loss
        self.ave_val_loss_cnt += 1
        
        return loss
        
        
    def test_step(self, batch, batch_idx):
        
        rays_data = batch
        rays_lum_target = rays_data[:, 6]
        rays_lum = self.forward(rays_data)
        
        loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        self.log('test_loss', loss, prog_bar=True)
        
        return loss
        
        
    def configure_optimizers(self):
        return self.setup_trainning()
    
    
    def view_eta_field(self, precision=32):
        x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3).to(self.device)
        
        with torch.no_grad():
            eta, _ = get_eta_manual(self, points)
        plot_3d(eta, precision)
    
    
    def on_train_start(self):
        self.view_eta_field()
    
    
    def on_train_epoch_end(self):
        
        self.log("ave_train_loss", self.ave_train_loss/self.ave_train_loss_cnt, prog_bar=True, sync_dist=True)
        self.ave_train_loss_cnt = 0
        self.ave_train_loss = 0.
        
        self.global_epoch += 1
        if self.view_per_epoch > 0 and self.global_epoch % self.view_per_epoch == 0:
            self.view_eta_field()
            print("\n")
            
        if self.save_per_epoch > 0 and self.global_epoch % self.save_per_epoch == 0:
            self.save_model(self.save_path)
            
            
    def on_validation_epoch_end(self):
        
        self.log("ave_val_loss", self.ave_val_loss/self.ave_val_loss_cnt, prog_bar=True, sync_dist=True)
        self.ave_val_loss_cnt = 0
        self.ave_val_loss = 0.
            
            
    # return optimizer list, configured in configure_optimizers
    def optimizers(self, use_pl_optimizer: bool = True):
        optimizers = super().optimizers(use_pl_optimizer)
        
        if isinstance(optimizers, list) is False:
            return [optimizers]
        
        return optimizers
    
    
    # return lr_scheduler list, configured in configure_optimizers
    def lr_schedulers(self):
        lr_schedulers = super().lr_schedulers()

        if lr_schedulers is None:
            return []
        
        if isinstance(lr_schedulers, list) is False:
            return [lr_schedulers]
        
        return lr_schedulers
            
            
    def setup_from_file(self, file_config: FileConfig):
        
        data_path = file_config.data_path
        if file_config.data_type == "pt":
            means = torch.load(os.path.join(data_path, "means.pt")).to(self.device)
            scales = torch.load(os.path.join(data_path, "scales.pt")).to(self.device)
            rotations = torch.load(os.path.join(data_path, "rotations.pt")).to(self.device)
            opacities = torch.load(os.path.join(data_path, "opacities.pt")).to(self.device)
        
        if not file_config.activated:
            scales = self.scale_inverse_activation(scales)
            rotations = self.rotation_inverse_activation(rotations)
            opacities = self.opacity_inverse_activation(torch.clamp(opacities, min=1e-6, max=1.0-1e-6))
        
        means = nn.Parameter(means, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        rotations = nn.Parameter(rotations, requires_grad=True)
        opacities = nn.Parameter(opacities, requires_grad=True)
        
        property_dict = {
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        self.n_gaussians = means.shape[0]
        self.set_properties(property_dict)
        
        
    def save_model(self, path):
        path = os.path.join(path, "epoch_{}".format(self.global_epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.means, os.path.join(path, "means.pt"))
        torch.save(self.scales, os.path.join(path, "scales.pt"))
        torch.save(self.rotations, os.path.join(path, "rotations.pt"))
        torch.save(self.opacities, os.path.join(path, "opacities.pt"))
            
            
    def setup_randomize(self, random_config: RandomizationConfig):
        
        self.n_gaussians = random_config.n_gaussians
        means = torch.rand([self.n_gaussians, 3], device=self.device, dtype=torch.float) * (random_config.means_rg[1] - random_config.means_rg[0]) + random_config.means_rg[0]
        scales = self.scale_inverse_activation(torch.rand([self.n_gaussians, 3], device=self.device, dtype=torch.float) * (random_config.scales_rg[1] - random_config.scales_rg[0]) + random_config.scales_rg[0])
        rotations = self.rotation_inverse_activation(torch.rand([self.n_gaussians, 4], device=self.device, dtype=torch.float) * (random_config.rotation_rg[1] - random_config.rotation_rg[0]) + random_config.rotation_rg[0])
        opacities = self.opacity_inverse_activation(torch.rand([self.n_gaussians], device=self.device, dtype=torch.float) * (random_config.opacities_rg[1] - random_config.opacities_rg[0]) + random_config.opacities_rg[0])
        
        means = nn.Parameter(means, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        rotations = nn.Parameter(rotations, requires_grad=True)
        opacities = nn.Parameter(opacities, requires_grad=True)
        
        property_dict = {
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        self.n_gaussians = means.shape[0]
        self.set_properties(property_dict)
        
        
    def setup_empty(self, n_gaussians: int = 1):
        
        self.n_gaussians = n_gaussians
        means = torch.zeros([self.n_gaussians, 3], device=self.device, dtype=torch.float)
        scales = torch.zeros([self.n_gaussians, 3], device=self.device, dtype=torch.float)
        rotations = torch.ones([self.n_gaussians, 4], device=self.device, dtype=torch.float)
        opacities = torch.zeros([self.n_gaussians], device=self.device, dtype=torch.float)
        
        means = nn.Parameter(means, requires_grad=True)
        scales = nn.Parameter(scales, requires_grad=True)
        rotations = nn.Parameter(rotations, requires_grad=True)
        opacities = nn.Parameter(opacities, requires_grad=True)
        
        property_dict = {
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        self.n_gaussians = means.shape[0]
        self.set_properties(property_dict)
        
            
    def set_properties(self, properties: dict[str, torch.Tensor]):
        self.properties = properties
        for name in self.names:
            self.gaussians[name] = properties[name]
            
            
    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(opacities)

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return torch.log(opacities / (1 - opacities))

    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales)

    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.log(scales)

    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(rotations)

    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        return rotations
    
    # these properties are activated, to get raw properties, using self.guassians['name']
    @property
    def scales(self):
        return self.scale_activation(self.gaussians["scales"])

    @property
    def rotations(self):
        return self.rotation_activation(self.gaussians["rotations"])

    @property
    def means(self):
        return self.gaussians["means"]

    @property
    def opacities(self):
        return self.opacity_activation(self.gaussians["opacities"])

    