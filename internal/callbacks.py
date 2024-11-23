import torch
import lightning.pytorch as pl
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F
import os 
from lightning.pytorch.callbacks.checkpoint import Checkpoint

class TestCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        print("Epoch is starting")

    def on_train_epoch_end(self, trainer, pl_module):
        print("Epoch is ending")
        
        
class LogCallback(pl.Callback):
    
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log_metrics_epoch()
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        pl_module.log_metrics_batch()
        