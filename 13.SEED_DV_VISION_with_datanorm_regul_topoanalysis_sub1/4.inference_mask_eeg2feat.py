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

def validation(epoch, Enc_eeg, val_dataloader, args):

    running_loss      = []
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])

    Enc_eeg.eval()
    # Proj_eeg.eval()

    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, labels, _) in enumerate(tq, start=1):
        eeg[:, :, args.mask_t_start:args.mask_t_end] = 0.0
        eeg, labels = eeg.to(args.device).unsqueeze(dim=1), labels.to(args.device)

        with torch.no_grad():
            x_proj = Enc_eeg(eeg)

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = args.num_classes
        
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)

    print("[Epoch: {}, Test KMeans score Proj: {}, Mask time range: {}-{}]".format(epoch, clustering_acc_proj, args.mask_t_start, args.mask_t_end))

    return clustering_acc_proj


def parse_args():
    parser = argparse.ArgumentParser('argument for EEG feature extraction')

    parser.add_argument('--test_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--exp_type', type=str, required=True, help='which EEG channel to use ["all", "left", "right"]')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--num_classes', type=int, default=15, help='total number of the classes for EEG dataset')
    parser.add_argument('--mask_t_start', type=int, required=True, help='Mask range start time')
    parser.add_argument('--mask_t_end', type=int, required=True, help='Mask range end time')
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
    test_dataset   = VideoEEGDataset(args.test_path,split_type='test', args=args)

    val_dataloader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('*'*50)
    print(f'Experiment type: {args.exp_type}, Total channels: {EEG_CAP_SHOT_V1[args.exp_type].shape[0]}')
    print('*'*50)

    Enc_eeg  = gat_network.EEGEncoder(eeg_channels=EEG_CAP_SHOT_V1[args.exp_type].shape[0], in_timesteps=args.eeg_timesteps, exp_type=args.exp_type).to(args.device)
    Enc_eeg.apply(gat_network.weights_init_normal)
    
    best_ckpt = []

    for path in natsorted(glob(args.ckpt_path+'/*')):
        acc = float(os.path.splitext(os.path.split(path)[-1])[0].split('_')[-1])*100 
        best_ckpt.append((path, acc))

    best_ckpt = sorted(best_ckpt, key=lambda x: x[1], reverse=True)


    ckpt_path = best_ckpt[0][0]
    print(f'Using checkpoint: {ckpt_path}')
    
    ######################################################
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    Enc_eeg.load_state_dict(checkpoint['enc_eeg_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    ######################################################
    
    val_acc   = validation(START_EPOCH, Enc_eeg, val_dataloader, args)