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


def train(epoch, Enc_eeg, optimizer, loss_fn, miner, train_dataloader, experiment_num, args):
        
    running_loss      = []
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])

    tq = tqdm(train_dataloader)

    Enc_eeg.train()
    # Proj_eeg.train()

    for batch_idx, (eeg, labels, _) in enumerate(tq, start=1):

        eeg    = eeg.to(args.device).unsqueeze(dim=1)
        labels = labels.to(args.device)
        optimizer.zero_grad()

        x_proj = Enc_eeg(eeg)
        # x_proj = Proj_eeg(x_proj)
        # x_proj = F.normalize(x_proj, p=2, dim=-1)

        hard_pairs = miner(x_proj, labels)
        loss       = loss_fn(x_proj, labels, hard_pairs)
        
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    # if (epoch%args.val_freq) == 0:
    #     Enc_eeg.eval()
    #     # Proj_eeg.eval()
        
    #     for batch_idx, (eeg, labels, _) in enumerate(tqdm(train_dataloader)):
    #         eeg, labels = eeg.to(args.device).unsqueeze(dim=1), labels.to(args.device)
    #         with torch.no_grad():
    #             x_proj = Enc_eeg(eeg)
    #             # x_proj = Proj_eeg(Enc_eeg(eeg))
    #             # x_proj = F.normalize(x_proj, p=2, dim=-1)

    #         eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
    #         labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    #     ### compute k-means score and Umap score on the text and image embeddings
    #     num_clusters   = args.num_classes

    #     k_means        = K_means(n_clusters=num_clusters)
    #     clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
        
    #     # neigh = KNeighborsClassifier(n_neighbors=5)
    #     # neigh.fit(eeg_featvec_proj, labels_array)
    #     # knn_score = neigh.score(eeg_featvec_proj, labels_array)
    #     # print("[Epoch: {}, Train KMeans score Proj: {}, Train KNN score: {}]".format(epoch, clustering_acc_proj, knn_score))
    #     print("[Epoch: {}, Train KMeans score Proj: {}]".format(epoch, clustering_acc_proj))


    return running_loss
 

def validation(epoch, Enc_eeg, optimizer, loss_fn, miner, val_dataloader, experiment_num, args):

    running_loss      = []
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])

    Enc_eeg.eval()
    # Proj_eeg.eval()

    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, labels, _) in enumerate(tq, start=1):
        eeg, labels = eeg.to(args.device).unsqueeze(dim=1), labels.to(args.device)

        with torch.no_grad():
            # x_proj = Proj_eeg(Enc_eeg(eeg))
            # x_proj = F.normalize(x_proj, p=2, dim=-1)
            x_proj = Enc_eeg(eeg)

            hard_pairs = miner(x_proj, labels)
            loss       = loss_fn(x_proj, labels, hard_pairs)

            running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = args.num_classes
        
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    
    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(eeg_featvec_proj, labels_array)
    # knn_score = neigh.score(eeg_featvec_proj, labels_array)
    # print("[Epoch: {}, Val KMeans score Proj: {}, Val KNN score: {}]".format(epoch, clustering_acc_proj, knn_score))
    print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))

    return running_loss, clustering_acc_proj


def parse_args():
    parser = argparse.ArgumentParser('argument for EEG feature extraction')

    parser.add_argument('--train_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--val_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--test_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--exp_type', type=str, required=True, help='which EEG channel to use ["all", "left", "right"]')
    # parser.add_argument('--split_type', type=str, required=True, help='channel split type to use ["normal", "random"]')
    parser.add_argument('--num_classes', type=int, default=15, help='total number of the classes for EEG dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size to use for training/inference')
    # parser.add_argument('--eeg_channels', type=int, default=62, help='number of eeg channels in the data')
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
    train_dataset  = VideoEEGDataset(args.train_path,split_type='train', args=args)
    val_dataset    = VideoEEGDataset(args.val_path,split_type='val', args=args)
    test_dataset   = VideoEEGDataset(args.test_path,split_type='test', args=args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('*'*50)
    print(f'Experiment type: {args.exp_type}, Total channels: {EEG_CAP_SHOT_V1[args.exp_type].shape[0]}')
    print('*'*50)

    Enc_eeg  = gat_network.EEGEncoder(eeg_channels=EEG_CAP_SHOT_V1[args.exp_type].shape[0], in_timesteps=args.eeg_timesteps, exp_type=args.exp_type).to(args.device)
    Enc_eeg.apply(gat_network.weights_init_normal)
    
    miner   = miners.MultiSimilarityMiner()
    loss_fn = losses.TripletMarginLoss()
    
    optimizer = optim.Adam(list(Enc_eeg.parameters()),\
                           lr=args.lr, weight_decay=1e-4)

    
    ######## Checkpoint Loading and Codebackup ########
    dir_info  = natsorted(glob('EXPERIMENT_EEG2FeatAbl_{}_*'.format(args.exp_type)))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    ckpt_lst = natsorted(glob('EXPERIMENT_EEG2FeatAbl_{}_{}/checkpoints/eegfeat_*.pth'.format(args.exp_type, experiment_num)))
    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        Enc_eeg.load_state_dict(checkpoint['enc_eeg_state_dict'])
        # Proj_eeg.load_state_dict(checkpoint['proj_eeg_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
        START_EPOCH += 1
    else:
        os.makedirs('EXPERIMENT_EEG2FeatAbl_{}_{}/checkpoints/'.format(args.exp_type, experiment_num))
        os.makedirs('EXPERIMENT_EEG2FeatAbl_{}_{}/bestckpt/'.format(args.exp_type, experiment_num))
        os.system('cp *.py EXPERIMENT_EEG2FeatAbl_{}_{}'.format(args.exp_type, experiment_num))
    ######################################################


    best_val_acc   = 0.0
    best_val_epoch = 0

    for epoch in range(START_EPOCH, args.epochs):

        running_train_loss = train(epoch, Enc_eeg, optimizer, loss_fn, miner, train_dataloader, experiment_num, args)
        if (epoch%args.val_freq) == 0:
            running_val_loss, val_acc   = validation(epoch, Enc_eeg, optimizer, loss_fn, miner, val_dataloader, experiment_num, args)

            if best_val_acc < val_acc:
                best_val_acc   = val_acc
                best_val_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'enc_eeg_state_dict': Enc_eeg.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'EXPERIMENT_EEG2FeatAbl_{}_{}/bestckpt/eegfeat_{}_{}.pth'.format(args.exp_type, experiment_num, 'all', val_acc))


        torch.save({
                'epoch': epoch,
                'enc_eeg_state_dict': Enc_eeg.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'EXPERIMENT_EEG2FeatAbl_{}_{}/checkpoints/eegfeat_{}.pth'.format(args.exp_type, experiment_num, 'all'))
