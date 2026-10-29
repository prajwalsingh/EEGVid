import torch
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_height = 128#224
image_width  = 128#224
input_channel= 3
kernel_size  = 3
padding      = 1
batch_size   = 32#256#256
num_workers  = 8
dataset_name = 'seed_video_eeg'
projection_dim = 1045