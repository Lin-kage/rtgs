# main.py
from lightning.pytorch.cli import LightningCLI
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
from internal.config import *
from internal.density_controller import DensityController
from internal.test import EtaDataLoader, EtaGaussianModel, EtaNerf
from internal.debug import tensor_grid_test

from internal.debug import print_tensor
import time


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    # cli_main()

    seed = 40
    torch.manual_seed(seed=seed)
    
    data_path = './data/matern_s8'
    
    lum_field = (np.load(data_path + '/lum_field.npy', allow_pickle=False))
    eta_true = (np.load(data_path + '/eta_true.npy', allow_pickle=True))
    
    plot_3d(torch.Tensor(eta_true), 16, reverse=True)
    # plot_3d(torch.tensor(lum_field), 64)
    
    trainer = pl.Trainer(max_epochs=500, accelerator='gpu', devices=[6])
    trainer.fit(
        GaussianModel(
            RenderConfig=RenderConfig(lum_field_fn=TensorGrid3D(torch.tensor(lum_field).reshape(64,64,64)).interp_linear,),
            OptimizationConfig=OptimizationConfig(means_lr=1e-4, opacities_lr=1e-4, scales_lr=1e-4, rotations_lr=1e-4),
            ViewConfig=ViewConfig(view_per_epoch=1, enable_view=True, save_per_epoch=10, save_path='./gaussian_save/gaussian1028'),
            FileConfig=FileConfig(data_path='./gaussian_save/gaussian1028/epoch_40', data_type='pt', activated=False),
            # RandomizationConfig=RandomizationConfig(n_gaussians=2000, scales_rg=[.05, .5], opacities_rg=[0.0, 0.0001]),
            density_controller=DensityController(
                densify_epoch_from_until=[0,1000], 
                densify_grad_threshold=1e-3,
                cull_opacity_threshold=[1e-7, 1e-2],
                cull_scale_threshold=[0.005, 0.5],
                clone_split_threshold=0.5,
                ),
            ),
        RaysDataLoader(data_path='./gaussian_save/test1', data_type='manual', batchsize=32)
        )
    
    # ## Create Data
    # data_path = './data/matern_s8'
    # lum_field = (np.load(data_path + '/lum_field.npy', allow_pickle=False))
    # eta_test = FieldGenerator(init_from_file='./gaussian_save/gaussian1.002_4/gaussian_35')
    # rays = torch.from_numpy(np.load(data_path + '/rays.npy', allow_pickle=False)).to("cuda")
    # rays_o = rays[:, :3]
    # rays_dir = rays[:, 3:6]
    # rays_lum = ray_trace(rays_o, rays_dir, d_s=1.2/64, d_steps=64, 
    #                      lum_field_fn=TensorGrid3D(torch.tensor(lum_field).reshape(64,64,64)).interp_linear,
    #                      eta_field_fn=lambda x: TensorGrid3D(torch.tensor(eta_true).reshape(16,16,16)).interp_linear(x), 
    #                      auto_grad=True
    #                      )
    # rays_data = torch.cat([rays_o, rays_dir, rays_lum[:, None]], dim=1)
    # # print(f"rays lum: {rays_lum}\n")
    # # torch.save(rays_data, './gaussian_save/test1/rays_data.pt')
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
    
    