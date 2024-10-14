import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
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
        
        etas_true = self.eta_field(batch[:, :3])
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('train_loss', loss, prog_bar=True)
                
        edge_etas, _ = get_eta_manual(self.gaussians, self.edge_points)
        edge_etas = torch.sum(torch.square(edge_etas - 1), dim=-1)
        
        return (1 - self.edge_fac) * loss + self.edge_fac * edge_etas
            
            
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
        
        self.veiw_etas()
        
        
    def on_train_start(self):
        
        self.veiw_etas()
    
        
    # def on_train_end(self):
    #     precision = 80
    #     x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
    #     points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
        
    #     eta, _ = get_eta_manual(self.gaussians, points)
        
    #     plot_3d(eta, precision, torch.zeros([1,3]))
        
    #     eta_true = self.eta_field(points)
        
    #     plot_3d(eta_true, precision, torch.zeros([1,3]))
        
    #     print("view end")
    
    
    def veiw_etas(self, precision=32):
        
        x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
        
        eta, _ = get_eta_manual(self.gaussians, points)
        
        plot_3d(eta, precision, torch.zeros([1,3]))
        
        eta_true = self.eta_field(points)
        
        plot_3d(eta_true, precision, torch.zeros([1,3]))
        
        print("view end")
    
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
class EtaDataLoader(pl.LightningDataModule):
    def __init__(self, data_path="", data_type="", batchsize=64, num=10000, using_rand=False, precision=64, eta_manual=None):
        super().__init__()
        self.data_path = data_path
        self.data_type = data_type
        self.batchsize = batchsize
        self.num = num
        self.using_rand = using_rand
        self.precision = precision
        if not using_rand:
            self.num = precision * precision * precision
        self.eta_manual = eta_manual
        
        
    def setup(self, stage : str):
        
        self.points = torch.empty([self.num, 4])
        
        if self.data_type == 'manual':
            etas_true = self.eta_manual
        
        if self.data_type == 'matern':
            
            etas_true = np.load(self.data_path + '/eta_true.npy', allow_pickle=False)
        
        interp = Grid3D(etas_true).interp            
            
        if self.using_rand:

            max_eta = np.max(etas_true)
        
            self.points[:, :3] = torch.rand([self.num, 3])
            
            # points_interp = interp(self.points[:, :3])
            # small_points = points_interp <= 1 + (max_eta - 1) * 0.5
            # small_points_cnt = torch.sum(small_points)
            # while small_points_cnt > 0.1 * self.num:
            #     self.points[small_points, :3] = torch.rand([small_points_cnt, 3])
            #     points_interp = interp(self.points[:, :3])
            #     small_points = points_interp <= 1 + (max_eta - 1) * 0.1
            #     small_points_cnt = torch.sum(small_points)
                
            # print(f"small / sum_points: {small_points_cnt} / {self.num}")
                
        else:
            x, y, z = torch.meshgrid(torch.linspace(0, 1, self.precision), 
                                        torch.linspace(0, 1, self.precision), 
                                        torch.linspace(0, 1, self.precision), indexing='xy')
            self.points[:, :3] = torch.stack([x, y, z], -1).reshape(-1,3)                
            
        
        self.points[:, 3] = interp(self.points[:, :3])
        
        # plot_3d(torch.tensor(self.points[:, 3]), self.precision)
            
    
    def train_dataloader(self):
        return DataLoader(self.points[:int(self.num)], batch_size=self.batchsize, shuffle=True)
        
        
    
    def val_dataloader(self):
        return DataLoader(self.points[int(.9*self.num):], batch_size=self.batchsize, shuffle=False)
    
        
    
    def test_dataloader(self):
        return DataLoader(self.points, batch_size=self.batchsize, shuffle=True)
        
        

class EtaNerf(pl.LightningModule):
    
    def __init__(self, trunk_depth = 8, trunk_width = 256, input_size=3, output_size = 1, 
            L_embed = 6, skips = [4], view_per_epoch = 10, lr = 1e-5, eta_field_fn = None, edge_precision=32):
        super().__init__()
        
        self.trunk_depth = trunk_depth
        self.trunk_width = trunk_width
        self.skips = skips
        self.input_size = input_size
        self.output_size = output_size
        self.view_per_epoch = view_per_epoch
        self.view_count = 0
        self.lr = lr
        self.eta_field = eta_field_fn
        self.L_embed = L_embed
        
        self.input_size = self.input_size + 2 * self.L_embed * self.input_size
        
        self.linears = torch.nn.ModuleList(
            [nn.Linear(self.input_size, self.trunk_width)] + 
            [(nn.Linear(self.trunk_width, self.trunk_width) if i not in self.skips 
             else nn.Linear(self.trunk_width + self.input_size, self.trunk_width)) for i in range(self.trunk_depth - 1)]
        )
        
        self.output_linear = nn.Linear(self.trunk_width, self.output_size)
        
        
    def forward(self, input):   
        
        input = self._embed(input[:, :3])
        # input = input[:, :3]
        
        input_change = input
        for i in range(len(self.linears)):
            
            input_change = self.linears[i](input_change)
            input_change = F.relu(input_change)
            if i in self.skips:
                input_change = torch.concat([input_change, input], -1)
                
        eta = self.output_linear(input_change)
        
        return eta
    
    
    def training_step(self, batch, batch_idx):
        
        # assert(False)
        
        etas_true = batch[:, 3]
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('train_loss', loss, prog_bar=True)
                
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        etas_true = batch[:, 3]
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('val_loss', loss, prog_bar=True)
                
        return loss
    
    
    def test_step(self, batch, batch_idx):
        
        etas_true = batch[:, 3]
        etas = self.forward(batch)
        
        loss = torch.mean(torch.square(etas - etas_true))
        self.log('test_loss', loss, prog_bar=True)
                
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
    def on_train_epoch_end(self):
        
        self.view_count += 1
        if self.view_count != self.view_per_epoch:
            return
        self.view_count = 0
        
        precision = 50
        x, y, z = torch.meshgrid(torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), torch.linspace(0, 1, precision), indexing='xy')
        points = torch.stack([x, y, z], -1).reshape(-1,3).to("cuda")
        
        with torch.no_grad():
            eta = self.forward(points) 
        
        plot_3d(eta, precision, torch.zeros([1,3]))
        
        if (self.eta_field is not None):
            eta_true = self.eta_field(points)
            
            plot_3d(eta_true, precision, torch.zeros([1,3]))
            
            # self.log('plot loss', torch.mean(torch.square(eta-eta_true)), prog_bar=True)
        
        print("view end")

    
    def _embed(self, x):
        rets = [x]
        for i in range(0, self.L_embed):
            for f in [torch.sin, torch.cos]:
                rets.append(f(2.0 ** i * x))
        
        return torch.cat(rets, dim=-1)