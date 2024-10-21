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
    
    # emplify the field
    # eta_true = (eta_true - 1) * 10 + 1
    
    # plot_3d(torch.Tensor(eta_true), 16, reverse=True)
    # plot_3d(torch.tensor(lum_field), 64)
    trainer = pl.Trainer(max_epochs=500, accelerator='gpu', devices=[7])
    trainer.fit(
        GaussianModel(lum_field_fn = TensorGrid3D(torch.from_numpy(lum_field).reshape(64,64,64)).interp_linear, 
                      d_steps=50, lr=1e-5, view_per_epochs=1,
                      n_gaussians=2000, init_randomlize=True,
                    #   gaussian_file='./gaussian_save/gaussian1.002/gaussian_3',
                      device='cuda'),
        RaysDataLoader(data_path='./gaussian_save/test1', data_type='manual', batchsize=32)
        )
    
    #### Create Data
    # data_path = './data/matern_s8'
    # lum_field = (np.load(data_path + '/lum_field.npy', allow_pickle=False))
    # eta_true = FieldGenerator(init_from_file='./gaussian_save/test1')
    # eta_test = FieldGenerator(init_from_file='./gaussian_save/gaussian1.002/gaussian_3')
    # rays = torch.from_numpy(np.load(data_path + '/rays.npy', allow_pickle=False)).to("cuda")
    # rays_o = rays[:, :3]
    # rays_dir = rays[:, 3:6]
    # rays_lum = ray_trace(rays_o, rays_dir, d_s=1.2/50, d_steps=50, 
    #                      lum_field_fn=TensorGrid3D(torch.tensor(lum_field).reshape(64,64,64)).interp_linear,
    #                      eta_field_fn=lambda x: eta_true.get_eta(x), 
    #                      auto_grad=False
    #                      )
    # rays_data = torch.cat([rays_o, rays_dir, rays_lum[:, None]], dim=1)
    # print(f"rays lum: {rays_lum}\n")
    # print(f"rays lum test: {rays_lum_test}\n")
    # test_loss = torch.mean(torch.square(rays_lum - rays_lum_test))
    # print(f"test loss: {test_loss}")
    # torch.save(rays_data, './gaussian_save/test1/rays_data.pt')
    
    #### test if gaussian can present the field 
    # test_trainer = pl.Trainer(max_epochs=150, accelerator='gpu', devices=[1], strategy='ddp_find_unused_parameters_true')
    # test_trainer.fit(
    #     # EtaNerf(trunk_depth=3,skips=[], trunk_width=256, view_per_epoch=25, lr=1e-3, L_embed=6, eta_field_fn=Grid3D(eta_true).interp),
    #     EtaGaussianModel(Grid3D(eta_true).interp, lr=2e-5, n_gaussians=2000, 
    #                      view_per_epoch=1, edge_fac=0, 
    #                      load_path='./gaussian_save/gaussian1.002/gaussian_3',
    #                      save_path='./gaussian_save/gaussian1.002_4', save_per_epoch=1),
    #     EtaDataLoader(data_type="manual", eta_manual=eta_true, batchsize=64, precision=32)
    # )
    
    
    # note: it is good practice to implement the CLI in a function and call it in the main if block
    
    