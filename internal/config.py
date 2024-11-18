import torch

class ViewConfig:
    def __init__(self, view_per_epoch = 10, enable_view = True, save_per_epoch = 10, save_path = None):
        self.view_per_epoch = view_per_epoch
        self.save_per_epoch = save_per_epoch
        self.save_path = save_path
        self.enable_view = enable_view
        

class RenderConfig:
    def __init__(self,
                 lum_field_fn = lambda _:0, 
                 d_steps: int = 64,
                 total_range: float = 1.2):
        self.lum_field_fn = lum_field_fn
        self.d_steps = d_steps
        self.total_range = total_range
        self.d_s = self.total_range / self.d_steps


class OptimizationConfig:
    def __init__(self, means_lr = 1e-4, scales_lr = 1e-4, rotations_lr = 1e-4, opacities_lr = 1e-4, reg_factor = 1e-4):
        self.means_lr = means_lr
        self.scales_lr = scales_lr
        self.rotations_lr = rotations_lr
        self.opacities_lr = opacities_lr
        self.reg_factor = reg_factor
        
        

class RandomizationConfig:
    def __init__(self, n_gaussians: int = 1, 
                 means_rg: tuple[float] = (.0, 1.0), 
                 scales_rg: tuple[float] = (.0, 1.0), 
                 rotation_rg: tuple[float] = (.0, 1.0), 
                 opacities_rg: tuple[float] = (.0, .01)):
        self.n_gaussians = n_gaussians
        self.means_rg = means_rg
        self.scales_rg = scales_rg
        self.rotation_rg = rotation_rg
        self.opacities_rg = opacities_rg
        


class FileConfig:
    def __init__(self, data_path: str, data_type: str, activated: bool = True):
        self.data_path = data_path
        self.data_type = data_type
        self.activated = activated
        
