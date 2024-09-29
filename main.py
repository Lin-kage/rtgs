# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

from internal.gaussian import Gaussian

import torch
import os
from internal.viewer import plot_3d
from internal.render import ray_trace
from internal.utils import get_eta_autograd

from internal.debug import print_tensor
import time


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    # cli_main()

    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

    seed = 2
    
    t = time.time()
    
    torch.manual_seed(seed=seed)
    
    gaussians = Gaussian(30)
    gaussians.init_randomize()
    precision = 80
    
    x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
    points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
    
    eta, d_eta = get_eta_autograd(gaussians, points)
    
    print(f"Calc Over, time cost {time.time() - t}")
    
    # print_tensor("mean", gaussians.means)
    # print_tensor("scale", gaussians.scales)
    # print_tensor("eta", eta)
    # print_tensor("d_eta", d_eta)
    
    t = time.time()
    
    test_rays_o = torch.tensor([[0.,.5 ,.5], [0.,.5 ,.6],[0.,.5 ,.7],[0.,.5 ,.4],[0.,.5 ,.3]]).to(gaussians.device)
    test_rays_dir = torch.tensor([[1.,0.,0],[1.,0.,0],[1.,0.,0],[1.,0.,0],[1.,0.,0] ]).to(gaussians.device)
    
    test_rays_points = ray_trace(gaussians, test_rays_o, test_rays_dir, 0.01, 150)
    
    # print_tensor("points", test_rays_points.transpose(0,1))

    print(f"Trace Over, time cost {time.time() - t}")
    
    plot_3d(eta, precision, test_rays_points)
    
    # note: it is good practice to implement the CLI in a function and call it in the main if block