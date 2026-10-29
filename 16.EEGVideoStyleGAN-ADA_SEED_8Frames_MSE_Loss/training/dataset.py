# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import config
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from videoeeg_dataset import VideoEEGDataset
import multiprocessing as mp
import pickle

try:
    import pyspng
except ImportError:
    pyspng = None

def encode_position_vectorized(x, enc_dim):
    
    # Create power vector: [0, 1, 2, ..., enc_dim-1]
    powers = torch.arange(enc_dim, device=x.device)
    
    # Calculate frequencies: [2^0, 2^1, 2^2, ..., 2^(enc_dim-1)]
    frequencies = torch.pow(2.0, powers)
    
    # Expand dimensions for broadcasting
    # x shape becomes compatible with frequencies
    x_expanded = x.unsqueeze(-1)
    
    # Multiply x by all frequencies at once
    xf = x_expanded * frequencies
    
    # Calculate sin and cos for all frequencies simultaneously
    sin_encodings = torch.sin(xf)
    cos_encodings = torch.cos(xf)
    
    # Interleave sin and cos values
    encodings = torch.stack([sin_encodings, cos_encodings], dim=-1).flatten(-2)
    
    # Concatenate original x with encodings along the last dimension
    result = torch.cat([x.unsqueeze(-1), encodings], dim=-1)
    
    return result


class EEG2ImageDataset(Dataset):
    def __init__(self, path, resolution=None, split_type='train', **super_kwargs):
        print(super_kwargs)
        self.dataset_path = path
        self.eegs   = []
        self.images_path = []
        self.labels = []
        self.images_idx = []
        self.class_name = []
        self.eeg_feat = []
        cls_lst = [0, 1]
        self._raw_shape = [3, config.image_height, config.image_width]
        self.resolution = config.image_height
        self.has_labels  = True
        self.label_shape = [config.projection_dim]
        self.label_dim   = config.projection_dim
        self.name        = config.dataset_name
        self.image_shape  = [3, config.image_height, config.image_width]
        self.num_channels = config.input_channel
        self.resize_transform = transforms.Resize((config.image_height, config.image_width))

        x_dataset  = VideoEEGDataset(self.dataset_path,split_type=split_type, load_frames=True)

        self.total_frames   = 8
        # print('loading dataset...')
        
        if os.path.exists('cache/eeg_feat.pkl'):
            with open('cache/eeg_feat.pkl', 'rb') as f:
                self.eeg_feat = pickle.load(f)
            with open('cache/labels.pkl', 'rb') as f:
                self.labels = pickle.load(f)
            with open('cache/images_path.pkl', 'rb') as f:
                self.images_path = pickle.load(f)
            with open('cache/images_idx.pkl', 'rb') as f:
                self.images_idx = pickle.load(f)
        
        else:
            os.makedirs('cache', exist_ok=True)

            # with mp.Pool(8) as pool:
            #     self.images = []
            #     self.eeg_feat = []
            #     self.labels = []
            #     print('Processing dataset...')
                
                # for i, results in enumerate(pool.apply_async(load_data, [(self.total_frames, self.resize_transform, eeg_feat.detach().cpu(), class_id, video_frames) for _, eeg_feat, class_id, _, video_frames, _ in x_dataset])):
                #     self.images.extend(results[0])
                #     self.eeg_feat.extend(results[1])
                #     self.labels.extend(results[2])
                #     sys.stderr.write('\rProcessing | Progress: {:4d}/{:4d}'.format(i,len(x_dataset)))    

            for _, eeg_feat, class_id, _, video_frames_path, _ in tqdm(x_dataset):
                eeg_feat = eeg_feat.detach().cpu()
                for per_frame_idx in range(self.total_frames):
                    # image = self.resize_transform(torch.clip((video_frames[:,per_frame_idx,:,:]*0.5+0.5)*255.0, 0.0, 255.0))
                    self.images_path.append(video_frames_path)
                    self.images_idx.append(per_frame_idx)
                    self.labels.append(class_id*self.total_frames+per_frame_idx)
                    temp_eeg_feat = encode_position_vectorized(torch.tensor(per_frame_idx).float(), 10)
                    self.eeg_feat.append(torch.cat([eeg_feat, temp_eeg_feat], dim=-1))

            # self.images   = torch.stack(self.images, dim=0)#.to(torch.float32)
            self.eeg_feat = torch.stack(self.eeg_feat, dim=0).to(torch.float32)
            self.labels   = torch.from_numpy(np.array(self.labels)).to(torch.int32)
            self.images_idx = torch.from_numpy(np.array(self.images_idx)).to(torch.int32)
            
            with open('cache/eeg_feat.pkl', 'wb') as f:
                pickle.dump(self.eeg_feat, f)
            with open('cache/labels.pkl', 'wb') as f:
                pickle.dump(self.labels, f)
            with open('cache/images_path.pkl', 'wb') as f:
                pickle.dump(self.images_path, f)
            with open('cache/images_idx.pkl', 'wb') as f:
                pickle.dump(self.images_idx, f)
                
        print('Dataset loaded and shape...')
        print(len(self.images_path), self.eeg_feat.shape, self.labels.shape, self.images_idx.shape)


    def __len__(self):
        return len(self.eeg_feat)

    def __getitem__(self, idx):
        video_data = torch.tensor(np.load(self.images_path[idx], allow_pickle=True)['frames']).float()
        image = self.resize_transform(torch.clip((video_data[:,self.images_idx[idx],:,:]*0.5+0.5)*255.0, 0.0, 255.0))
        # image = self.images[idx]
        label = self.labels[idx]
        con   = self.eeg_feat[idx]

        return image, con
    
    def get_label(self, idx):
        con = self.eeg_feat[idx]
        return con