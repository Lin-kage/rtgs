from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl
import numpy as np  
import os


class RaysDataLoader(pl.LightningDataModule):
    def __init__(self, data_path, data_type, batchsize=64, train_slice=0.95):
        super().__init__()
        self.data_path = data_path
        self.data_type = data_type
        self.batchsize = batchsize
        self.train_slice = train_slice
        
        
    def setup(self, stage : str):
        
        if self.data_type == 'matern':
            self.rays = np.load(self.data_path + '/rays.npy', allow_pickle=False)
            self.ray_lum_target = np.load(self.data_path + '/ray_lum_target.npy', allow_pickle=False)
            
            self.rays[:, :7] = self.ray_lum_target
            
            self.rays = torch.from_numpy(self.rays)
        elif self.data_type == 'manual':
            self.rays = torch.load(os.path.join(self.data_path, 'rays_data.pt'))
            
        self.rays = self.rays[torch.randperm(self.rays.shape[0]), :]
        if isinstance(self.train_slice, float):
            self.train_slice = int(self.rays.shape[0] * self.train_slice)
            
    
    def train_dataloader(self):
        return DataLoader(self.rays[:self.train_slice], batch_size=self.batchsize, shuffle=True)
        
        
    
    def val_dataloader(self):
        return DataLoader(self.rays[self.train_slice:], batch_size=self.batchsize, shuffle=False)
    
        
    
    def test_dataloader(self):
        return DataLoader(self.rays, batch_size=self.batchsize, shuffle=True)
        