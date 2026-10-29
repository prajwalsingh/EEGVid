import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from videoeeg_dataset import VideoEEGDataset, EEG_CAP_SHOT_V1  # Import your custom dataset
from tqdm import tqdm
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import datetime
from metrics import K_means
from pytorch_metric_learning import miners, losses
from sklearn.neighbors import KNeighborsClassifier
from natsort import natsorted
from glob import glob
import argparse

seed_n = 3407
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

import gat_network

def generate_embedding(Enc_eeg, x_dataloader, data_type, args):

    running_loss      = []
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])
    filename_array    = np.array([])

    Enc_eeg.eval()
    # Proj_eeg.eval()

    tq = tqdm(x_dataloader)
    for batch_idx, (eeg, labels, _, filename) in enumerate(tq, start=1):
        eeg, labels = eeg.to(args.device).unsqueeze(dim=1), labels.to(args.device)
        filename = np.array([item for item in filename], dtype=object)

        with torch.no_grad():
            x_proj = Enc_eeg(eeg)

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()
        filename_array    = np.concatenate((filename_array, filename), axis=0) if filename_array.size else filename

    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = args.num_classes
        
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)

    print("[{} KMeans score Proj: {}]".format(data_type, clustering_acc_proj))

    old_data_path = "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/{}/".format(data_type)
    new_data_path = "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200_featvec/eeg/{}/".format(data_type)
    os.makedirs(new_data_path, exist_ok=True)
    for eeg_featvec, label, file_id in zip(tqdm(eeg_featvec_proj), labels_array, filename_array):
        eeg = np.load(os.path.join(old_data_path, f'{file_id}.npy'), allow_pickle=True)
        data = {}
        data['eeg'] = eeg
        data['eeg_featvec'] = eeg_featvec
        np.save(os.path.join(new_data_path, f'{file_id}.npy'), data)

    return clustering_acc_proj


def parse_args():
    parser = argparse.ArgumentParser('argument for EEG feature extraction')

    parser.add_argument('--test_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--exp_type', type=str, required=True, help='which EEG channel to use ["all", "left", "right"]')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--num_classes', type=int, default=15, help='total number of the classes for EEG dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size to use for training/inference')
    parser.add_argument('--eeg_timesteps', type=int, default=400, help='number of timesteps in eeg sample')
    parser.add_argument('--feat_dims', type=int, default=1024, help='eeg feature size as output')
    parser.add_argument('--epochs', type=int, default=300, help='total epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--val_freq', type=int, default=10, help='run validation after epoch n')
    parser.add_argument('--device', type=str, default='cuda', help='gpu')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for pytorch dataloader')
    
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    
    args = parse_args()

    train_root_dir = "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/train/"
    val_root_dir   = "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/val/"
    test_root_dir  = "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/test/"

    train_dataset = VideoEEGDataset(train_root_dir,split_type='train', args=args, return_file_name=True)
    val_dataset   = VideoEEGDataset(val_root_dir,split_type='val', args=args, return_file_name=True)
    test_dataset  = VideoEEGDataset(test_root_dir,split_type='test', args=args, return_file_name=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('*'*50)
    print(f'Experiment type: {args.exp_type}, Total channels: {EEG_CAP_SHOT_V1[args.exp_type].shape[0]}')
    print('*'*50)

    Enc_eeg  = gat_network.EEGEncoder(eeg_channels=EEG_CAP_SHOT_V1[args.exp_type].shape[0], in_timesteps=args.eeg_timesteps, exp_type=args.exp_type).to(args.device)
    Enc_eeg.apply(gat_network.weights_init_normal)
    
    # print(natsorted(glob(args.ckpt_path+'/*')))
    # import pdb; pdb.set_trace()

    ckpt_path = natsorted(glob(args.ckpt_path+'/*'))[-1]
    ######################################################
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    Enc_eeg.load_state_dict(checkpoint['enc_eeg_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    ######################################################
    
    generate_embedding(Enc_eeg, train_dataloader, data_type='train', args=args)
    generate_embedding(Enc_eeg, val_dataloader, data_type='val', args=args)
    generate_embedding(Enc_eeg, test_dataloader, data_type='test', args=args)