
class RenderConfig:
    def __init__(self, lum_field_path, sum_steps, d_steps, d_s):
        self.lum_field_path = lum_field_path
        self.sum_steps = sum_steps
        self.d_steps = d_steps
        self.d_s = d_s
        

class SetupConfig:
    def __init__(self, setup_option, random_n, random_means, random_scales, random_rotation, random_opacities, from_file_path, from_file_type, activated):
        self.setup_option = setup_option
        self.random_n = random_n
        self.means_rg = random_means
        self.scales_rg = random_scales
        self.rotation_rg = random_rotation
        self.opacities_rg = random_opacities
        
        self.from_file_path = from_file_path
        self.from_file_type = from_file_type
        self.activated = activated
        

class OptimizationConfig:
    def __init__(self, automatic_optimization, means_lr, scales_lr, rotations_lr, opacities_lr, reg_factor):
        self.automatic_optimization = automatic_optimization
        self.means_lr = means_lr
        self.scales_lr = scales_lr
        self.rotations_lr = rotations_lr
        self.opacities_lr = opacities_lr
        self.reg_factor = reg_factor
        
        
class ViewConfig:
    def __init__(self, enable_view, view_per_epoch):
        self.enable_view = enable_view
        self.view_per_epoch = view_per_epoch
        

class CheckpointConfig:
    def __init__(self, enable_checkpoint, checkpoint_path, checkpoint_per_epoch):
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_path = checkpoint_path
        self.checkpoint_per_epoch = checkpoint_per_epoch