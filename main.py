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
from internal.field import Grid3D, TensorGrid3D
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
    
    lum_field = (np.load('./data/matern_s8/lum_field.npy', allow_pickle=False))
    eta_true = (np.load('./data/matern_s8/eta_true.npy', allow_pickle=True))
    
    # test_trainer = pl.Trainer(max_epochs=150, accelerator='gpu', devices=[7], strategy='ddp_find_unused_parameters_true')
    # test_trainer.fit(
    #     # EtaNerf(trunk_depth=8,skips=[4], view_per_epoch=25, lr=1e-5, eta_field_fn=Grid3D((eta_true-1)*100).interp),
    #     EtaGaussianModel(Grid3D((eta_true-1)*100 + 1).interp, lr=1e-4, n_gaussians=1000, view_per_epoch=5, edge_fac=5e-4),
    #     EtaDataLoader(data_path='./data/matern_s8', data_type='matern', batchsize=512, precision=32)
    # )
    
    # trainer = pl.Trainer(max_epochs=500, accelerator='gpu', devices=[7])
    # trainer.fit(
    #     GaussianModel(TensorGrid3D(torch.from_numpy(lum_field).reshape(64,64,64)).interp_linear, d_steps=100),
    #     # GaussianModel(Grid3D(lum_field.reshape(64,64,64)).interp),
    #     RaysDataLoader(data_path='./data/matern_s8', data_type='matern')
    #     )
    
    # t = time.time()
    # gaussians = Gaussian(1)
    # gaussians.init_randomize()
    # precision = 80
    
    # x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
    # points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
    
    # # eta, d_eta = get_eta_autograd(gaussians, points)
    # eta, d_eta = get_eta_manual(gaussians, points)
    
    # print(f"Calc Over, time cost {time.time() - t}")
    
    # # print_tensor("mean", gaussians.means)
    # # print_tensor("scale", gaussians.scales)
    # # print_tensor("eta", eta)
    # # print_tensor("d_eta", d_eta)
    
    
    # test_rays_o = torch.tensor([[0.,.5 ,.5], [0.,.5 ,.6],[0.,.5 ,.7],[0.,.5 ,.4],[0.,.5 ,.3]]).to(gaussians.device)
    # test_rays_dir = torch.tensor([[1.,0.,0],[1.,0.,0],[1.,0.,0],[1.,0.,0],[1.,0.,0] ]).to(gaussians.device)
    
    # test_rays_points = ray_trace(gaussians, test_rays_o, test_rays_dir, 0.01, 150, auto_grad=False)
    
    # # print_tensor("points", test_rays_points.transpose(0,1))

    # print(f"Trace Over, time cost {time.time() - t}")
    
    # plot_3d(eta, precision, test_rays_points)    
    
    # note: it is good practice to implement the CLI in a function and call it in the main if block
    
    