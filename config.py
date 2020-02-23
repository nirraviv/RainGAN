import torch
from pathlib import Path

# needed train or just generate images
is_train = True
load_from_iter = None

# SimGAN (True) or RainGAN (False)
is_eyes = False

# name of the folder to store all logs (tb, weights, images etc.)
experiment_name = Path('rain/vgg_temp2')

# Is the PC has cuda
cuda_use = torch.cuda.is_available()

# learning rate for D, the lr in Apple blog is 0.0001
d_lr = 0.000001
# learning rate for R, the lr in Apple blog is 0.0001
r_lr = 0.000001
# lambda in paper, the author of the paper said it's 0.01
delta = 0.003  # TODO: need to find the correct value

# image dimensions
img_width = 55
img_height = 35
img_channels = 1

# get synthetic and real image path
if is_eyes:
    syn_path = Path('dataset/gaze.npy')
    real_path = Path('dataset/real_gaze.npy')
else:
    syn_path = Path('dataset/syn_rain.npy')
    real_path = Path('dataset/real_rain.npy')

# result show in 4 triplet (synthetic, refined, real) samples per line
pics_line = 4

# =================== training params ======================
# pre-train R times
r_pretrain = 500
# pre-train D times
d_pretrain = 100
# train steps
train_steps = 100000

batch_size = 128
test_batch_size = 128
# the history buffer size
buffer_size = 12800
k_d = 1  # number of discriminator updates per step
k_r = 30 # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 50
# output D pre-training result per times
d_pre_per = 50
# save model dictionary and training dataset output result per train times
save_per = 10 # 100 (Currently save just to tensorboard)

# pre-training dictionary path
# ref_pre_path = 'R _pre.pkl'
ref_pre_path = None
# disc_pre_path = 'D_pre.pkl'
disc_pre_path = None

# dictionary saving path
models_path = Path('logs') / experiment_name / 'models'
D_path = 'D_{0}.pkl'
R_path = 'R_{0}.pkl'

# training result path to save
train_res_path = Path('logs') / experiment_name / 'train_res'

# create dirs
train_res_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)
