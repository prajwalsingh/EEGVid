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
    def __init__(self,root_dir, split_type='train', args=None, channel_num=None):
        self.root_dir = root_dir
        self.split_type = split_type
        self.args = args
        self.files = natsorted(glob(os.path.join(self.root_dir, '*.npy')))
        print(f'Found {len(self.files)} files in {self.root_dir}')

        if self.args.exp_type == 'per_channel':
            self.filter_channels = channel_num - 1
        else:
            self.filter_channels = EEG_CAP_SHOT_V1[self.args.exp_type] - 1
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        eeg = np.load(current_file, allow_pickle=True)
        
        subject_id = trial_id = int(os.path.splitext(os.path.basename(current_file))[0].split('_')[0])
        trial_id = int(os.path.splitext(os.path.basename(current_file))[0].split('_')[1])
        
        eeg = eeg[self.filter_channels]

        ### Scale: -1 to 1 ####
        max_norm = np.max(eeg)
        min_norm = np.min(eeg)
        eeg = 2 * (eeg - min_norm) / (max_norm - min_norm) - 1
        ########################

        return torch.tensor(eeg).float(), torch.tensor(trial_id).long(), torch.tensor(subject_id).long()

        
    
    
if __name__ == '__main__':
    train_root_dir = '../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/train'
    val_root_dir = '../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/val'
    test_root_dir = '../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test'
    train_dataset = VideoEEGDataset(train_root_dir,split_type='train')
    val_dataset = VideoEEGDataset(val_root_dir,split_type='val')
    test_dataset = VideoEEGDataset(test_root_dir,split_type='test')
    next(iter(train_dataset))
    # BATCH_SIZE = 256
    # NUM_WORKERS = 16
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers =NUM_WORKERS,pin_memory=False)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers =NUM_WORKERS,pin_memory=False)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers =NUM_WORKERS,pin_memory=False)
    # for eeg, text_emb, trial_id, emotion in tqdm(train_dataloader):
    #     pass
        # print(eeg.shape)
        # print(text_emb.shape)
        # print(trial_id.shape)
        # print(emotion.shape)
