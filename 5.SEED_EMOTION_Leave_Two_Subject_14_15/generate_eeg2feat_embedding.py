import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from videoeeg_dataset import VideoEEGDataset  # Import your custom dataset
from tqdm import tqdm
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import datetime
from metrics import K_means, TsnePlot

seed_n = 3407
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

import gat_network
import config

def generate_embedding(Enc_eeg, data_loader, data_type, repeat_num=0):
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])
    file_id_array     = np.array([])

    for batch_idx, (eeg, _, _, labels, _, file_id) in enumerate(tqdm(data_loader)):
        eeg, labels = eeg.to(config.device).unsqueeze(dim=1), labels.to(config.device)
        file_id = np.array([item for item in file_id], dtype=object)
        with torch.no_grad():
            x_proj = Enc_eeg(eeg)
            # x_proj = Proj_eeg(Enc_eeg(eeg))
            # x_proj = F.normalize(x_proj, p=2, dim=-1)
        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()
        file_id_array     = np.concatenate((file_id_array, file_id), axis=0) if file_id_array.size else file_id



    k_means = K_means(n_clusters=15, random_state=45)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)

    print(f"{data_type} Acc: {clustering_acc_proj:.15f}")

    old_data_path = f'../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_{repeat_num}/{data_type}/'
    
    new_data_path = f'../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens_eegfeat/repeat_{repeat_num}/{data_type}/'
    os.makedirs(new_data_path, exist_ok=True)

    for eeg_featvec, label, file_id in zip(tqdm(eeg_featvec_proj), labels_array, file_id_array):
        data = np.load(os.path.join(old_data_path, f'{file_id}.npy'), allow_pickle=True).item()
        data['eeg_featvec'] = eeg_featvec
        np.save(os.path.join(new_data_path, f'{file_id}.npy'), data)
    # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    # reduced_embedding = tsne_plot.plot(eeg_featvec_proj, labels_array, clustering_acc_proj, 3, data_type='Test')



if __name__ == '__main__':
    # Create a unique folder for this run
    # run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # ckpt_dir = os.path.join('ckpts', run_id)
    # os.makedirs(ckpt_dir, exist_ok=True)
    # log_file = open(os.path.join(ckpt_dir, 'log.txt'), 'w')

    # Hyperparameters
    input_size = 62
    output_size = 300
    learning_rate = 3e-4
    epochs = 300
    BATCH_SIZE = 128#16
    NUM_WORKERS = 16
    early_stopping_patience = 20

    repeat_num = '*'
    # Dataloaders
    train_root_dir = '../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_{}/train'.format(repeat_num)
    val_root_dir = '../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_{}/val'.format(repeat_num)
    test_root_dir = '../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_{}/test'.format(repeat_num)
    train_dataset = VideoEEGDataset(train_root_dir,split_type='train', req_file_id=True)
    val_dataset = VideoEEGDataset(val_root_dir,split_type='val', req_file_id=True)
    test_dataset = VideoEEGDataset(test_root_dir,split_type='test', req_file_id=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Instantiate model, loss function, and optimizer
    # model = WavenetRegressor(input_size, output_size).cuda()
    # model.apply(init_weights)  # Initialize weights

    Enc_eeg  = gat_network.EEGEncoder(eeg_channels=config.num_channels, in_timesteps=config.time_steps).to(config.device)

    ckpt_path = 'EXPERIMENT_EEG2Feat_3/bestckpt/eegfeat_all_0.7126797385620915.pth'
    # ckpt_path = 'EXPERIMENT_EEG2Feat_4/bestckpt/eegfeat_all_0.5959477124183007.pth'
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    Enc_eeg.load_state_dict(checkpoint['enc_eeg_state_dict'])
    Enc_eeg.eval()
    
    generate_embedding(Enc_eeg, train_dataloader, data_type='train', repeat_num=repeat_num)
    generate_embedding(Enc_eeg, val_dataloader, data_type='val', repeat_num=repeat_num)
    generate_embedding(Enc_eeg, test_dataloader, data_type='test', repeat_num=repeat_num)