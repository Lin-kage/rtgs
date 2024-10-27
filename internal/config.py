
class ViewConfig:
    def __init__(self, view_per_epoch = 10):
        self.view_per_epoch = view_per_epoch
        

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
    means_lr: float = 1e-4
    scales_lr: float = 1e-4
    rotations_lr: float = 1e-4
    opacities_lr: float = 1e-4
    

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
    def __init__(self, data_path: str, data_type: str):
        self.data_path = data_path
        self.data_type = data_type
