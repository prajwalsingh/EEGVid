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
    def __init__(self, eeg_data, eeg_label, eeg_sub_label, args):
    
        self.eeg_data  = eeg_data
        self.eeg_label = eeg_label
        self.eeg_sub_label = eeg_sub_label
        self.args = args

        # if self.args.exp_type == 'per_channel':
        #     self.filter_channels = channel_num - 1
        # else:
        #     self.filter_channels = EEG_CAP_SHOT_V1[self.args.exp_type] - 1
        
    def __len__(self):
        return len(self.eeg_label)
    
    def gaussian_noise(self, eeg_data, std_dev=0.2):
        """
        Adds Gaussian noise to the EEG data.
        
        Parameters:
        - eeg_data: EEG data tensor of shape (128, 440)
        - std_dev: Standard deviation of the noise distribution
        
        Returns:
        - Noisy EEG data

        Best result we get so far at std_dev = 0.2
        """
        noise = torch.randn_like(eeg_data) * std_dev
        return eeg_data + noise

    def time_reversal(self, eeg_data, prob=0.3):
        """
        Reverses the time axis of the EEG data with a given probability.
        
        Parameters:
        - eeg_data: EEG data tensor of shape (128, 440)
        - prob: Probability of reversing the time axis
        
        Returns:
        - Time-reversed EEG data if probability condition is met
        """
        if np.random.rand() < prob:
            return torch.fliplr(eeg_data)
        else:
            return eeg_data

    def time_masking(self, eeg_data, mask_length=10):
        """
        Masks a portion of the time series data.
        
        Parameters:
        - eeg_data: EEG data tensor of shape (128, 440)
        - mask_length: Length of the time mask
        
        Returns:
        - Time-masked EEG data
        """
        mask_start = np.random.randint(0, eeg_data.shape[1] - mask_length)
        mask_end = mask_start + mask_length
        eeg_data[:, mask_start:mask_end] = 0
        return eeg_data



    def __getitem__(self, idx):
        
        eeg           = self.eeg_data[idx]
        eeg_label     = self.eeg_label[idx] - 1
        eeg_sub_label = self.eeg_sub_label[idx] - 1

        if (self.args is not None) and (self.args.data_aug is not None):
            if 'gauss' in self.args.data_aug:
                # print('applying gauss augmn')
                eeg = self.gaussian_noise(eeg, std_dev=0.0)
            if 'timerev' in self.args.data_aug:
                # print('applying timerev augmn')
                eeg = self.time_reversal(eeg, prob=0.2)
            if 'timemask' in self.args.data_aug:
                # print('applying timemask augmn')
                eeg = self.time_masking(eeg, mask_length=10)

        # eeg = eeg[self.filter_channels]

        ### Scale: -1 to 1 ####
        max_norm = torch.max(eeg)
        min_norm = torch.min(eeg)
        eeg = 2 * (eeg - min_norm) / (max_norm - min_norm) - 1
        ########################

        # if self.return_file_name:
        #     return torch.tensor(eeg).float(), torch.tensor(trial_id).long(), torch.tensor(subject_id).long(), filename

        return eeg.float(), eeg_label.long(), eeg_sub_label.long()

        
    
    
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
