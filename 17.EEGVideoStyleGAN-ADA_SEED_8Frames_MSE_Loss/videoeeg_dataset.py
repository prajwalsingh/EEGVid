import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm


EEG_CAP_SHOT_V1 = {
            'all': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             60, 61, 62, 63, 64], dtype=np.int32),
            'noseback_left': np.array([1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19,
                                       24, 23, 22, 21, 20, 25, 26, 27], dtype=np.int32),
            'noseback_center': np.array([33, 37, 38, 47, 48, 32, 31, 30, 29, 28], dtype=np.int32),
            'noseback_right': np.array([34, 36, 35, 39, 40, 41, 42, 46, 45, 44, 43, 49, 50, 51, 52, 56, 55,
                                        54, 53, 57, 58, 59, 60, 61, 63, 62, 64], dtype=np.int32),
            'noseback_Q1_right': np.array([34, 36, 35, 39, 40, 41, 42, 46, 45, 44, 43], dtype=np.int32),
            'noseback_Q2_left': np.array([1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11], dtype=np.int32),
            'noseback_Q3_left': np.array([16, 17, 18, 19, 24, 23, 22, 21, 20, 25, 26, 27], dtype=np.int32),
            'noseback_Q4_right': np.array([56, 55, 54, 53, 57, 58, 59, 60, 61, 63, 62, 64], dtype=np.int32),
            'leftrightear_center': np.array([15, 14, 13, 12, 48, 49, 50, 51, 52], dtype=np.int32),
            'noseback_front': np.array([1, 33, 34, 2, 3, 37, 36, 35, 7, 6, 5, 4, 38, 39, 40, 41, 42,
                                        8, 9, 10, 11, 47, 46, 45, 44, 43], dtype=np.int32),
            'noseback_back': np.array([16, 17, 18, 19, 32, 56, 55, 54, 53, 24, 23, 22, 21, 20, 31, 57, 
                                       58, 59, 60, 61, 25, 26, 30, 63, 62, 27, 29, 64, 28], dtype=np.int32),
            'lobes_frontal_left': np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11], dtype=np.int32),
            'lobes_frontal_right': np.array([34, 35, 36, 39, 40, 41, 42, 46, 45, 44], dtype=np.int32),
            'lobes_parietal_left': np.array([17, 18, 19, 24, 23, 22, 21, 20], dtype=np.int32),
            'lobes_parietal_right': np.array([56, 55, 54, 57, 58, 59, 60, 61], dtype=np.int32),
            'lobes_occipital_left': np.array([25, 26, 27], dtype=np.int32),
            'lobes_occipital_right': np.array([63, 62, 64], dtype=np.int32),
            'lobes_temporal_left': np.array([8, 15, 16], dtype=np.int32),
            'lobes_temporal_right': np.array([43, 52, 53], dtype=np.int32),
            'lobes_frontal': np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 34, 35, 36, 39, 40, 41, 42, 46, 45, 44, 33, 37, 38, 47], dtype=np.int32),
            'lobes_parietal': np.array([17, 18, 19, 24, 23, 22, 21, 20, 56, 55, 54, 57, 58, 59, 60, 61, 32, 31], dtype=np.int32),
            'lobes_occipital': np.array([25, 26, 27, 63, 62, 64, 30, 29], dtype=np.int32),
            'lobes_temporal': np.array([8, 15, 16, 43, 52, 53], dtype=np.int32),
        }


class VideoEEGDataset(Dataset):
    def __init__(self,root_dir, split_type='train', args=None, channel_num=None, return_file_name=False, load_frames=False):
        self.root_dir = root_dir
        self.split_type = split_type
        self.args = args
        self.return_file_name = return_file_name
        self.load_frames = load_frames
        self.video_latent_path = '../dataset/Preprocessed_SingleShot/video_frames/'
        self.files = natsorted(glob(os.path.join(self.root_dir, '*.npy')))
        print(f'Found {len(self.files)} files in {self.root_dir}')

        self.filter_channels = EEG_CAP_SHOT_V1['all'] - 1
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        eeg_data = np.load(current_file, allow_pickle=True).item()
        eeg_feat = eeg_data['eeg_featvec']
        filename = os.path.splitext(os.path.basename(current_file))[0]

        # subject_id = int(filename.split('_')[0])
        chunk_id = int(filename.split('_')[2])
        trial_id = int(filename.split('_')[1])
        
        # eeg = eeg[self.filter_channels]

        # ### Scale: -1 to 1 ####
        # max_norm = np.max(eeg)
        # min_norm = np.min(eeg)
        # eeg = 2 * (eeg - min_norm) / (max_norm - min_norm) - 1
        # ########################

        # if self.load_frames:
        videoname = f'{trial_id}_{chunk_id}'
        # video_data = np.load(os.path.join(self.video_latent_path, videoname+'.npy'), allow_pickle=True).item()
        # video_frames = video_data['frames']
        
        video_frames_path = os.path.join(self.video_latent_path, videoname+'.npz')
        
        # return torch.tensor(eeg_feat).float(), torch.tensor(trial_id).long(), torch.tensor(video_frames).float(), filename
        return torch.tensor(eeg_feat).float(), torch.tensor(trial_id).long(), video_frames_path, filename

        # if self.return_file_name:
        #     return torch.tensor(eeg).float(), torch.tensor(trial_id).long(), torch.tensor(subject_id).long(), filename

        # return torch.tensor(eeg).float(), torch.tensor(trial_id).long(), torch.tensor(subject_id).long()

