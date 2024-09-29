from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl
import numpy as np  


class RaysDataLoader(pl.LightningDataModule):
    def __init__(self, data_path, data_type):
        super().__init__()
        self.data_path = data_path
        self.data_type = data_type
        
        
    def setup(self, stage : str):
        
        if self.data_type == 'matern':
            self.rays = np.load(self.data_path + '/rays.npy', allow_pickle=False)
            self.ray_lum_target = np.load(self.data_path + '/ray_lum_target.npy', allow_pickle=False)
            
            self.rays[:, :7] = self.ray_lum_target
            
            self.rays = torch.from_numpy(self.rays)
            
    
    def train_dataloader(self):
        return DataLoader(self.rays[:4000], shuffle=True)
        
        
    
    def val_dataloader(self):
        return DataLoader(self.rays[4000:], shuffle=False)
    
        
    
    def test_dataloader(self):
        return DataLoader(self.rays, shuffle=True)
        