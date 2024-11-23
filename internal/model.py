from typing import List, Optional, Union
import torch
import lightning.pytorch as pl
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
import os 
from .utils import get_eta_manual, get_eta_only
from .viewer import plot_3d
from .field import TensorGrid3D
from .config_class import *
from .density_controller import DensityController


class GaussianModel(pl.LightningModule):    

    def __init__(self,
                 renderConfig: RenderConfig,
                 optimizationConfig: OptimizationConfig,
                 viewConfig: ViewConfig,
                 setupConfig: SetupConfig,
                 density_controller = DensityController()
                 ):
        super().__init__()
        
        self.gaussians = nn.ParameterDict()
        
        self.names = ["means", "scales", "rotations", "opacities"]
        
        self.optimization_config = optimizationConfig
        self.automatic_optimization = optimizationConfig.automatic_optimization
        
        self.lum_field_path = renderConfig.lum_field_path
        self.d_steps = renderConfig.d_steps
        self.d_s = renderConfig.d_s
        
        self.setup_config = setupConfig
            
        self.view_config = viewConfig
        
        self.density_controller = density_controller
        
        
    def setup(self, stage: str):
        
        lum_field = (np.load(self.lum_field_path, allow_pickle=False))
        self.lum_field_fn = TensorGrid3D(grid_val=torch.tensor(lum_field).reshape(64,64,64), cval=.0).interp_linear
        
        if self.setup_config.setup_option == 2:
            self.setup_from_file(self.setup_config.from_file_path, self.setup_config.from_file_type, self.setup_config.activated)
        elif self.setup_config.setup_option == 1:
            self.setup_randomize(self.setup_config)
        elif self.setup_config.setup_option == 0:
            self.setup_empty(100)
        else:
            raise NotImplementedError
        
        self.ave_val_loss = 0
        self.ave_train_loss_cnt = 0
        self.ave_train_loss = 0
        self.ave_val_loss_cnt = 0
        self.global_epoch = 0
        
        # regularization    
        res = 24
        x, y, z = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing='xy')
        points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        select_mask = torch.logical_or(torch.any(points == 0, dim=-1), torch.any(points == 1, dim=-1))
        self.reg_points = points[select_mask]
        
        self.view_eta_field()
            
        
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
        if self.reg_points.device != self.device:
            self.reg_points = self.reg_points.to(self.device)
        
        self.regs = self.optimization_config.reg_factor * torch.sum(get_eta_only(self, self.reg_points)-1)
        self.loss = torch.mean(torch.square(rays_lum - rays_lum_target))
        
        self.ave_train_loss += self.loss
        self.ave_train_loss_cnt += 1
        
        # backward
        self.manual_backward(self.loss)
        
        means_grad = self.means.grad
        
        self.manual_backward(self.regs)
        
        new_properties, prune_mask, clone_mask = self.density_controller.before_step(means_grad, self, self.global_epoch)   
        
        for optimizer in optimizers:
            optimizer.step()
            
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()
        
        # if clone, each opacity is divided by 2
        with torch.no_grad():
            pad_mask = torch.zeros_like(self.gaussians['opacities'], dtype=torch.bool)
            pad_mask[:clone_mask.shape[0]] = clone_mask
            self.gaussians['opacities'][pad_mask] = self.opacity_inverse_activation(self.opacities[pad_mask] / 2.2)
        self.density_controller.after_step(new_properties, prune_mask, self, optimizers, self.global_epoch)
            
            
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
        print("configure_optimizers called")
        return self.setup_trainning()
    
    
    def view_eta_field(self, precision=32):
        if self.view_config.enable_view is False:
            return
        
        x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3).to(self.device)
        
        with torch.no_grad():
            eta, _ = get_eta_manual(self, points)
        plot_3d(eta, precision)

            
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
    
    
    def log_metrics_batch(self):
        self.log("loss", self.loss, prog_bar=True)
        self.log("n_gs", torch.tensor(self.n_gaussians, dtype=torch.float32), prog_bar=True)
        self.log("regs", self.regs, prog_bar=True)
        
        
    def log_metrics_epoch(self):
        self.log("ave_train_loss", self.ave_train_loss/self.ave_train_loss_cnt, prog_bar=True)
        self.log("ave_val_loss", self.ave_val_loss/self.ave_val_loss_cnt, prog_bar=True)
        self.ave_train_loss_cnt = 0
        self.ave_train_loss = 0.
        self.ave_val_loss_cnt = 0
        self.ave_val_loss = 0.
            
            
    def setup_from_file(self, data_path, data_type, activated):
        
        if data_type == "pt":
            means = torch.load(os.path.join(data_path, "means.pt")).to(self.device)
            scales = torch.load(os.path.join(data_path, "scales.pt")).to(self.device)
            rotations = torch.load(os.path.join(data_path, "rotations.pt")).to(self.device)
            opacities = torch.load(os.path.join(data_path, "opacities.pt")).to(self.device)
        if data_type == "ckpt":
            data = torch.load(data_path)
            means = data['state_dict']["gaussian.means"].to(self.device)
            scales = data['state_dict']["gaussian.scales"].to(self.device)
            rotations = data['state_dict']["gaussian.rotations"].to(self.device)
            opacities = data['state_dict']["gaussian.opacities"].to(self.device)
        
        if not activated:
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
            
            
    def setup_randomize(self, random_config: SetupConfig):
        
        self.n_gaussians = random_config.random_n
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
        for name in self.names:
            self.gaussians[name] = properties[name]
        self.n_gaussians = self.gaussians["means"].shape[0]
            
            
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
    
    # raw properties, not activated
    @property
    def properties(self):
        return {name: self.gaussians[name] for name in self.names}

    