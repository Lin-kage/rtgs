# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

from internal.gaussian import GaussianModel

import torch

from internal.utils import plot_3d


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    # cli_main()

    seed = 2
    
    torch.manual_seed(seed=seed)
    
    model = GaussianModel(10)
    precision = 64
    
    x, y, z = torch.meshgrid(torch.linspace(0, 5, precision), torch.linspace(0, 5, precision), torch.linspace(0, 5, precision), indexing='xy')
    points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
    
    eta, d_eta = model.get_eta(points)
    
    print("Calc Over")
    
    plot_3d(eta, precision)
    
    test_rays_o = torch.tensor([[0.,2.5,0.], [1.,0.,0.]])
    test_rays_dir = torch.tensor([[1.,0.,0.], [0.,1.,0.]])
    
    test_rays_points = model.ray_trace(test_rays_o, test_rays_dir, 0.05, 100)
    
    print(test_rays_points)
    
    # note: it is good practice to implement the CLI in a function and call it in the main if block