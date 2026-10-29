import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm


SEEDV_CAP = {
            'all': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             60, 61], dtype=np.int32),
            'noseback_left': np.array([1, 4, 6, 7, 8, 9, 15, 16, 17, 18, 24, 25, 26, 27, 33, 34, 35, 36, 
                                       42, 43, 44, 45, 51, 52, 53, 58, 59], dtype=np.int32),
            'noseback_center': np.array([2, 10, 19, 28, 37, 46, 54, 60], dtype=np.int32),
            'noseback_right': np.array([3, 5, 11, 12, 13, 14, 20, 21, 22, 23, 29, 30, 31, 32, 38, 39, 40,
                                        41, 47, 48, 49, 50, 55, 56, 57, 61, 62], dtype=np.int32),
            'noseback_Q1_right': np.array([3, 5, 11, 12, 13, 14, 20, 21, 22, 23], dtype=np.int32),
            'noseback_Q2_left': np.array([1, 3, 6, 7, 8, 9, 15, 16, 17, 18], dtype=np.int32),
            'noseback_Q3_left': np.array([33, 34, 35, 36, 42, 43, 44, 45, 51, 52, 53, 56, 59], dtype=np.int32),
            'noseback_Q4_right': np.array([38, 39, 40, 41, 47, 48, 49, 50, 55, 56, 57, 61, 62], dtype=np.int32),
            'leftrightear_center': np.array([24, 25, 26, 27, 28, 29, 30, 31, 32], dtype=np.int32),
            'noseback_front': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                                        19, 20, 21, 22, 23], dtype=np.int32),
            'noseback_back': np.array([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], dtype=np.int32),
            'lobes_frontal_left': np.array([1, 4, 6, 7, 8, 9, 16, 17, 18], dtype=np.int32),
            'lobes_frontal_right': np.array([3, 5, 11, 12, 13, 14, 20, 21, 22], dtype=np.int32),
            'lobes_parietal_left': np.array([34, 35, 36, 42, 43, 44, 45], dtype=np.int32),
            'lobes_parietal_right': np.array([38, 39, 40, 47, 48, 49, 50], dtype=np.int32),
            'lobes_occipital_left': np.array([51, 52, 53, 54], dtype=np.int32),
            'lobes_occipital_right': np.array([55, 56, 57, 61], dtype=np.int32),
            'lobes_temporal_left': np.array([15, 29, 33], dtype=np.int32),
            'lobes_temporal_right': np.array([23, 32, 41], dtype=np.int32),
            'lobes_frontal': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
                                       19, 20, 21, 22], dtype=np.int32),
            'lobes_parietal': np.array([34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=np.int32),
            'lobes_occipital': np.array([51, 52, 53, 54, 55, 56, 57, 59, 60, 61], dtype=np.int32),
            'lobes_temporal': np.array([15, 29, 33, 23, 32, 41], dtype=np.int32),
        }


class VideoEEGDataset(Dataset):
    def __init__(self,root_dir, split_type='train', args=None, channel_num=None, return_file_name=False, load_frames=False):
        self.root_dir = root_dir
        self.split_type = split_type
        self.args = args
        self.return_file_name = return_file_name
        self.load_frames = load_frames
        self.video_latent_path = '../dataset/SEED_DV_EEG_Features_sub1/video_frames_final/'
        self.files = natsorted(glob(os.path.join(self.root_dir, '*.npz')))
        print(f'Found {len(self.files)} files in {self.root_dir}')

        self.filter_channels = SEEDV_CAP['all'] - 1
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        eeg_data = np.load(current_file, allow_pickle=True)
        eeg_feat = eeg_data['eeg_feat']
        class_id = eeg_data['eeg_label']
        videoname = str(eeg_data['video_name'])
        video_frames_path = os.path.join(self.video_latent_path, videoname+'.npz')
        
        return torch.tensor(eeg_feat).float(), torch.tensor(class_id).long(), video_frames_path, videoname