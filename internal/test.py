import torch
import lightning.pytorch as pl
import numpy as np  
from .gaussian import Gaussian
from .render import ray_trace
from .utils import get_eta_autograd, get_eta_manual
from .field import Grid3D
from torch.utils.data import DataLoader
from .viewer import plot_3d

class EtaGaussianModel(pl.LightningModule):    

    def __init__(self, 
                 eta_field_fn, 
                 lr = 1e-5, n_gaussians = 100, view_per_epoch=5, 
                 edge_fac=.2, edge_presision=80):
        super().__init__()
        
        self.lr = lr
        self.edge_fac = edge_fac
        self.n_gaussians = n_gaussians
        self.eta_field = eta_field_fn
        self.view_count = 0
        self.view_per_epoch = view_per_epoch
        
        self.gaussians = Gaussian(n_gaussians)
        self.gaussians.init_randomize()
        
        x, y, z = torch.meshgrid(torch.linspace(0, 1, edge_presision), torch.linspace(0, 1, edge_presision), torch.linspace(0, 1, edge_presision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3)
        edge_points_select = torch.any(points==0, dim=1) | torch.any(points==1, dim=1)
        self.edge_points = points[edge_points_select]
        
        self.to(self.gaussians.means.device)
        
        
    def setup(self, stage:str):
        self.edge_points = self.edge_points.to(self.gaussians.device)
        
        
    def forward(self, input):   
        
        points = input[:, :3]

        etas, _ = get_eta_manual(self.gaussians, points)
            
        return etas

    
    def training_step(self, batch, batch_idx):
        
        etas_true = batch[:, 3]
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('train_loss', loss, prog_bar=True)
                
        edge_etas, _ = get_eta_manual(self.gaussians, self.edge_points)
        edge_etas = torch.sum(torch.square(edge_etas - 1), dim=-1)
        
        return (1 - self.edge_fac) * loss + self.edge_fac * (edge_etas - 1)
            
            
    def validation_step(self, batch, batch_idx):
        
        etas_true = batch[:, 3]
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
        
    def test_step(self, batch, batch_idx):
        
        etas_true = batch[:, 3]
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
    
    def on_train_epoch_end(self):
        
        self.view_count += 1
        if self.view_count != self.view_per_epoch:
            return
        self.view_count = 0
        
        precision = 50
        x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
        
        eta, _ = get_eta_manual(self.gaussians, points)
        
        plot_3d(eta, precision, torch.zeros([1,3]))
        
        eta_true = self.eta_field(points)
        
        plot_3d(eta_true, precision, torch.zeros([1,3]))
        
        print("view end")
    
        
    # def on_train_end(self):
    #     precision = 80
    #     x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
    #     points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
        
    #     eta, _ = get_eta_manual(self.gaussians, points)
        
    #     plot_3d(eta, precision, torch.zeros([1,3]))
        
    #     eta_true = self.eta_field(points)
        
    #     plot_3d(eta_true, precision, torch.zeros([1,3]))
        
    #     print("view end")
    
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
class EtaDataLoader(pl.LightningDataModule):
    def __init__(self, data_path, data_type, batchsize, num):
        super().__init__()
        self.data_path = data_path
        self.data_type = data_type
        self.batchsize = batchsize
        self.num = num
        
        
    def setup(self, stage : str):
        
        if self.data_type == 'matern':
            etas_true = np.load(self.data_path + '/eta_true.npy', allow_pickle=False)
            interp = Grid3D(etas_true).interp
            
            self.points = torch.empty([self.num, 4])
            self.points[:, :3] = torch.rand([self.num, 3])
            self.points[:, 3] = interp(self.points[:, :3])
            
            
    
    def train_dataloader(self):
        return DataLoader(self.points[:int(.9*self.num)], batch_size=self.batchsize, shuffle=True)
        
        
    
    def val_dataloader(self):
        return DataLoader(self.points[int(.9*self.num):], batch_size=self.batchsize, shuffle=False)
    
        
    
    def test_dataloader(self):
        return DataLoader(self.points, batch_size=self.batchsize, shuffle=True)
        