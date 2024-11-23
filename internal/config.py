#### General config
seed = 40
max_epochs = 100
accelerator = 'gpu'
devices = [6,7]

data_path = "./data/test4096"
data_type = "manual"
batchsize = 64
train_slice = 0.95

#### Render config
lum_field_path = "./data/lum_field.npy"
sum_steps = 1.2
d_steps = 64
d_s = sum_steps / d_steps

#### Opimization config
automatic_optimization = False
means_lr = 2e-4
scales_lr = 1e-4
rotations_lr = 1e-4
opacities_lr = 1e-5
reg_factor = 1e-7
# for density_controller
densify_epoch_from_until = [1000,1000]
densify_grad_threshold = 5e-4
cull_opacity_threshold = [5e-6, 1e-2]
cull_scale_threshold = [0., 1.0]
clone_split_threshold = 0.4

#### Setup config
# 0: empty  1: random  2: from file
setup_option = 1  

random_n = 4000
random_means = (.0, 1.0)
random_scales = (.05, .7)
random_rotations = (.0, 1.0)
random_opacities = (.0, 1e-5)

from_file_path = "./gaussian_save/gaussian1118/epoch_10"
from_file_type = "pt"
activated = True

# View config
enable_view = False
view_per_epoch = 10

# Checkpoint config
enable_checkpoint = True
checkpoint_path = "./gaussian_save/gaussian1123_test"
checkpoint_per_epoch = 1
