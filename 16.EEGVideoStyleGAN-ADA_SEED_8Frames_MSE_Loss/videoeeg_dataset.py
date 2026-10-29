import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm

class VideoEEGDataset(Dataset):
    def __init__(self,root_dir, split_type='train', load_frames=False, req_file_id=False):
        self.root_dir = root_dir
        self.split_type = split_type
        self.load_frames = load_frames
        self.req_file_id = req_file_id
        self.video_latent_path = '../dataset/SEED/data/video_frames/'
        self.files = natsorted(glob(os.path.join(self.root_dir, '*.npy')))
        print(f'Found {len(self.files)} files in {self.root_dir}')
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        file_id  = os.path.splitext(os.path.basename(current_file))[0]
        eeg_data = np.load(current_file, allow_pickle=True).item()
        
        eeg = eeg_data['eeg']
        eeg_feat = eeg_data['eeg_featvec']
        text_token_with_mask = eeg_data['video_text_token_with_mask']
        # tokens = text_token_with_mask[0].squeeze(axis=0)
        # mask   = text_token_with_mask[1].squeeze(axis=0)
        video_name = os.path.splitext(eeg_data['video_name'])[0]

        # if self.load_frames:
        # video_data = np.load(os.path.join(self.video_latent_path, video_name+'.npz'), allow_pickle=True)#.item()
        # # video_latent = video_data['video_encoding']
        # video_frames = video_data['frames']
        video_frames_path = os.path.join(self.video_latent_path, video_name+'.npz')
        # else:
        #     video_latent = np.load(os.path.join(self.video_latent_path, video_name+'.npz'), allow_pickle=True).item()['video_encoding']
        
        # norm_tokens   = np.zeros_like(tokens, dtype=np.float32)
        # non_zero_len  = mask.sum()
        # norm_tokens[:non_zero_len] = (tokens[:non_zero_len] / self.max_dict_len)

        ### Scale: -1 to 1 ####
        max_norm = np.max(eeg)
        min_norm = np.min(eeg)
        eeg = 2 * (eeg - min_norm) / (max_norm - min_norm) - 1
        ########################

        trial_id = eeg_data['trial_id']-1
        emotion = eeg_data['emotion']
        
        # if self.load_frames:
        # return torch.tensor(eeg).float(), torch.tensor(eeg_feat).float(), torch.tensor(trial_id).long(), torch.tensor(emotion).long(), torch.tensor(video_frames).float(), file_id
        return torch.tensor(eeg).float(), torch.tensor(eeg_feat).float(), torch.tensor(trial_id).long(), torch.tensor(emotion).long(), video_frames_path, file_id
        # else:
        #     return torch.tensor(eeg).float(), torch.tensor(eeg_feat).float(), torch.tensor(trial_id).long(), torch.tensor(emotion).long(), torch.tensor(video_latent).float()

        
    
    
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
