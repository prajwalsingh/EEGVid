import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm


EEG_CAP_SEED_V1 = {
            'all': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                             22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             60, 61, 62], dtype=np.int32),
            'noseback_left': np.array([1, 4, 6, 7, 8, 9, 15, 16, 17, 18, 24, 25, 26, 27, 33, 34, 35, 36,
                                       42, 43, 44, 45, 51, 52, 53, 58, 59], dtype=np.int32),
            'noseback_center': np.array([2, 10, 19, 28, 37, 46, 54, 60], dtype=np.int32),
            'noseback_right': np.array([3, 5, 11, 12, 13, 14, 20, 21, 22, 23, 29, 30, 31, 32, 38, 39, 40, 41,
                                        47, 48, 49, 50, 55, 56, 57, 61, 62], dtype=np.int32),
            'noseback_Q1_right': np.array([3, 5, 11, 12, 13, 14, 20, 21, 22, 23], dtype=np.int32),
            'noseback_Q2_left': np.array([1, 4, 6, 7, 8, 9, 15, 16, 17, 18], dtype=np.int32),
            'noseback_Q3_left': np.array([33, 34, 35, 36, 42, 43, 44, 45, 51, 52, 53, 58, 59], dtype=np.int32),
            'noseback_Q4_right': np.array([38, 39, 40, 41, 47, 48, 49, 50, 55, 56, 57, 61, 62], dtype=np.int32),
            'leftrightear_center': np.array([24, 25, 26, 27, 28, 29, 30, 31, 32], dtype=np.int32),
            'noseback_front': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                        20, 21, 22, 23], dtype=np.int32),
            'noseback_back': np.array([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], dtype=np.int32),
            'lobes_frontal_left': np.array([1, 4, 6, 7, 8, 9, 16, 17, 18], dtype=np.int32),
            'lobes_frontal_right': np.array([3, 5, 11, 12, 13, 14, 20, 21, 22], dtype=np.int32),
            'lobes_parietal_left': np.array([34, 35, 36, 42, 43, 44, 45], dtype=np.int32),
            'lobes_parietal_right': np.array([36, 39, 40, 47, 48, 49, 50], dtype=np.int32),
            'lobes_occipital_left': np.array([51, 52, 53, 59], dtype=np.int32),
            'lobes_occipital_right': np.array([55, 56, 57, 61], dtype=np.int32),
            'lobes_temporal_left': np.array([15, 24, 33], dtype=np.int32),
            'lobes_temporal_right': np.array([23, 32, 41], dtype=np.int32),
            'lobes_frontal': np.array([1, 4, 6, 7, 8, 9, 16, 17, 18, 3, 5, 11, 12, 13, 14, 20, 21, 22, 2, 10, 19], dtype=np.int32),
            'lobes_parietal': np.array([34, 35, 36, 42, 43, 44, 45, 36, 39, 40, 47, 48, 49, 50, 37, 46], dtype=np.int32),
            'lobes_occipital': np.array([51, 52, 53, 59, 55, 56, 57, 61, 54, 60], dtype=np.int32),
            'lobes_temporal': np.array([15, 23, 24, 32, 33, 41], dtype=np.int32),
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
            self.filter_channels = EEG_CAP_SEED_V1[self.args.exp_type] - 1
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        eeg_data = np.load(current_file, allow_pickle=True).item()
        eeg = eeg_data['eeg']
        trial_id = eeg_data['trial_id']-1
        emotion = eeg_data['emotion']
        # text_token_with_mask = eeg_data['video_text_token_with_mask']
        # tokens = text_token_with_mask[0].squeeze(axis=0)
        # mask   = text_token_with_mask[1].squeeze(axis=0)

        # norm_tokens   = np.zeros_like(tokens, dtype=np.float32)
        # non_zero_len  = mask.sum()
        # norm_tokens[:non_zero_len] = (tokens[:non_zero_len] / self.max_dict_len)
        eeg = eeg[self.filter_channels]

        ### Scale: -1 to 1 ####
        max_norm = np.max(eeg)
        min_norm = np.min(eeg)
        eeg = 2 * (eeg - min_norm) / (max_norm - min_norm) - 1
        ########################

        # return torch.tensor(eeg).float(), torch.tensor(tokens).float(), torch.tensor(mask).int(), torch.tensor(trial_id).long(), torch.tensor(emotion).long()
        return torch.tensor(eeg).float(), torch.tensor(trial_id).long(), torch.tensor(emotion).long()

        
    
    
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
