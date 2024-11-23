# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

import torch
import os
import numpy as np  

from internal.dataloader import RaysDataLoader
from internal.gaussian import Gaussian
from internal.model import GaussianModel
from internal.field import Grid3D, TensorGrid3D, FieldGenerator
from internal.viewer import plot_3d
from internal.render import ray_trace
from internal.utils import get_eta_autograd, get_eta_manual
from internal.config_class import *
from internal.density_controller import DensityController
from internal.test import EtaDataLoader, EtaGaussianModel, EtaNerf
from internal.debug import tensor_grid_test
from internal.callbacks import *

from internal.debug import print_tensor
import time


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    # cli_main()
    import internal.config as config
    
    torch.manual_seed(seed=config.seed)
    
    model_ckpt = ModelCheckpoint(
        dirpath=config.checkpoint_path,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=config.checkpoint_per_epoch,
        save_weights_only=True
    )
    
    trainer = pl.Trainer(max_epochs=config.max_epochs, accelerator=config.accelerator, devices=config.devices,
                         callbacks=[LogCallback(), model_ckpt],
                         strategy='auto'
                         )
    trainer.fit(
        GaussianModel(
            renderConfig=RenderConfig(config.lum_field_path, config.sum_steps, config.d_steps, config.d_s),
            optimizationConfig=OptimizationConfig(config.automatic_optimization, config.means_lr, config.scales_lr, config.rotations_lr, config.opacities_lr, config.reg_factor),
            viewConfig=ViewConfig(config.enable_view, config.view_per_epoch),
            setupConfig=SetupConfig(config.setup_option, config.random_n, config.random_means, config.random_scales, config.random_rotations, config.random_opacities, config.from_file_path, config.from_file_type, config.activated),
            density_controller=DensityController(config.densify_epoch_from_until, config.densify_grad_threshold, config.cull_opacity_threshold, config.cull_scale_threshold, config.clone_split_threshold),
            ),
        RaysDataLoader(data_path=config.data_path, data_type=config.data_type, batchsize=config.batchsize, train_slice=config.train_slice)
        )
    
    
    # data_path = './data/matern_s8'
    
    # lum_field = (np.load(data_path + '/lum_field.npy', allow_pickle=False))
    # eta_true = (np.load(data_path + '/eta_true.npy', allow_pickle=True))
    
    # plot_3d(torch.Tensor(eta_true), 16, reverse=True)
    # plot_3d(torch.tensor(lum_field), 64)
    
    # ## Create Data
    # data_path = './data/matern_s8'
    # lum_field = (np.load(data_path + '/lum_field.npy', allow_pickle=False))
    # # eta_true = (np.load(data_path + '/eta_true.npy', allow_pickle=True))
    # eta_true = FieldGenerator(init_from_file='./gaussian_save/gaussian1.002_4/gaussian_35')
    # rays = torch.from_numpy(np.load(data_path + '/rays.npy', allow_pickle=False)).to("cuda")
    # rays_o = rays[:, :3]
    # rays_dir = rays[:, 3:6]
    
    # n = 16000
    # x = torch.linspace(0+1e-5, 1-1e-5, n)
    # y = torch.linspace(0+1e-5, 1-1e-5, n)
    # z = torch.zeros(n)
    # rays_o = torch.stack([x,y,z], dim=1).to("cuda")
    # rays_dir = torch.tensor([[0.,0.,1.]]).repeat(n,1).to("cuda")
    
    # d_step = 100
    # rays_lum = ray_trace(rays_o, rays_dir, d_s=1.2/d_step, d_steps=d_step, 
    #                      lum_field_fn=TensorGrid3D(torch.tensor(lum_field).reshape(64,64,64)).interp_linear,
    #                     #  eta_field_fn=lambda x: TensorGrid3D(torch.tensor(eta_true).reshape(16,16,16)).interp_linear(x),
    #                      eta_field_fn=lambda x: eta_true.get_eta(x),
    #                      auto_grad=False
    #                      )
    # rays_data = torch.cat([rays_o, rays_dir, rays_lum[:, None]], dim=1)
    # print(f"rays lum: {rays_lum}\n")
    # torch.save(rays_data, './gaussian_save/test4096/rays_data.pt')
    # rays_lum_test = ray_trace(rays_o, rays_dir, d_s=1.2/64, d_steps=64, 
    #                      lum_field_fn=TensorGrid3D(torch.tensor(lum_field).reshape(64,64,64)).interp_linear,
    #                      eta_field_fn=lambda x: eta_test.get_eta(x), 
    #                      auto_grad=False
    #                      )
    # test_loss = torch.mean((rays_lum_test - rays_lum)**2)
    # print(f"test loss: {test_loss}")
    
    # ### test if gaussian can present the field 
    # test_trainer = pl.Trainer(max_epochs=150, accelerator='gpu', devices=[1], strategy='ddp_find_unused_parameters_true')
    # test_trainer.fit(
    #     # EtaNerf(trunk_depth=3,skips=[], trunk_width=256, view_per_epoch=25, lr=1e-3, L_embed=6, eta_field_fn=Grid3D(eta_true).interp),
    #     EtaGaussianModel(Grid3D(eta_true).interp, lr=2e-5, n_gaussians=2000, 
    #                      view_per_epoch=1, edge_fac=0, 
    #                      load_path=None,
    #                      save_path='./gaussian_save/gaussian1.002_4', save_per_epoch=1),
    #     EtaDataLoader(data_type="manual", eta_manual=eta_true, batchsize=64, precision=32)
    # )
    
    
    # note: it is good practice to implement the CLI in a function and call it in the main if block
    
    