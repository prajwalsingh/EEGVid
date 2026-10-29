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


chosed_label = [i for i in range(1,41)]
GT_label = np.array([[23, 22, 9, 6, 18, 14, 5, 36, 25, 19, 28, 35, 3, 16, 24, 40, 15, 27, 38, 33,
                      34, 4, 39, 17, 1, 26, 20, 29, 13, 32, 37, 2, 11, 12, 30, 31, 8, 21, 7, 10, ],
                     [27, 33, 22, 28, 31, 12, 38, 4, 18, 17, 35, 39, 40, 5, 24, 32, 15, 13, 2, 16,
                      34, 25, 19, 30, 23, 3, 8, 29, 7, 20, 11, 14, 37, 6, 21, 1, 10, 36, 26, 9, ],
                     [15, 36, 31, 1, 34, 3, 37, 12, 4, 5, 21, 24, 14, 16, 39, 20, 28, 29, 18, 32,
                      2, 27, 8, 19, 13, 10, 30, 40, 17, 26, 11, 9, 33, 25, 35, 7, 38, 22, 23, 6, ],
                     [16, 28, 23, 1, 39, 10, 35, 14, 19, 27, 37, 31, 5, 18, 11, 25, 29, 13, 20, 24,
                      7, 34, 26, 4, 40, 12, 8, 22, 21, 30, 17, 2, 38, 9, 3, 36, 33, 6, 32, 15, ],
                     [18, 29, 7, 35, 22, 19, 12, 36, 8, 15, 28, 1, 34, 23, 20, 13, 37, 9, 16, 30,
                      2, 33, 27, 21, 14, 38, 10, 17, 31, 3, 24, 39, 11, 32, 4, 25, 40, 5, 26, 6, ],
                     [29, 16, 1, 22, 34, 39, 24, 10, 8, 35, 27, 31, 23, 17, 2, 15, 25, 40, 3, 36,
                      26, 6, 14, 37, 9, 12, 19, 30, 5, 28, 32, 4, 13, 18, 21, 20, 7, 11, 33, 38],
                     [38, 34, 40, 10, 28, 7, 1, 37, 22, 9, 16, 5, 12, 36, 20, 30, 6, 15, 35, 2,
                      31, 26, 18, 24, 8, 3, 23, 19, 14, 13, 21, 4, 25, 11, 32, 17, 39, 29, 33, 27]
                     ])

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

    parser.add_argument('--eeg_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--subject', type=str, default=None, help='subject id')
    parser.add_argument('--data_aug', nargs='*', default=None, help="['gauss', 'timerev','timemask']")
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