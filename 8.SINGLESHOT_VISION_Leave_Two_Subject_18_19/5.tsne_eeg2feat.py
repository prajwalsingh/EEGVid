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
from metrics import K_means, TsnePlot
from pytorch_metric_learning import miners, losses
from sklearn.neighbors import KNeighborsClassifier
from natsort import natsorted
from glob import glob
import argparse
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import style
import seaborn as sns
import pandas as pd

style.use('ggplot')

seed_n = 3407
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

import gat_network

def validation(epoch, Enc_eeg, val_dataloader, args):

    running_loss         = []
    eeg_featvec_proj     = np.array([])
    labels_array         = np.array([])
    subject_array         = np.array([])

    Enc_eeg.eval()
    # Proj_eeg.eval()

    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, labels, subject_id) in enumerate(tq, start=1):
        eeg, labels = eeg.to(args.device).unsqueeze(dim=1), labels.to(args.device)
        subject_id  = subject_id.to(args.device)

        with torch.no_grad():
            x_proj = Enc_eeg(eeg)

        eeg_featvec_proj     = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array         = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()
        subject_array         = np.concatenate((subject_array, subject_id.cpu().detach().numpy()), axis=0) if subject_array.size else subject_id.cpu().detach().numpy()

    return eeg_featvec_proj, labels_array, subject_array


def parse_args():
    parser = argparse.ArgumentParser('argument for EEG feature extraction')

    parser.add_argument('--test_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--save_path', type=str, default=None, help='save path for tsne plot')
    parser.add_argument('--exp_type', type=str, required=True, help='which EEG channel to use ["all", "left", "right"]')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--vid_labels', type=int, default=15, help='total number of the video classes for EEG dataset')
    parser.add_argument('--sub_labels', type=int, default=19, help='total number of the subject classes for EEG dataset')
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
    

    ######################################################
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    Enc_eeg.load_state_dict(checkpoint['enc_eeg_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    ######################################################
    
    eeg_feat_vec, eeg_feat_vid_label, eeg_feat_sub_label  = validation(START_EPOCH, Enc_eeg, val_dataloader, args)

    ### compute k-means score and Umap score on the text and image embeddings
    k_means        = K_means(n_clusters=args.vid_labels)
    clustering_acc_proj_vid_label = k_means.transform(eeg_feat_vec, eeg_feat_vid_label)
    print('KMeans clustering score for video label train on video label: {}'.format(clustering_acc_proj_vid_label))

    k_means        = K_means(n_clusters=args.vid_labels)
    clustering_acc_proj_sub_label = k_means.transform(eeg_feat_vec, eeg_feat_sub_label)
    print('KMeans clustering score for video label train on video label: {}'.format(clustering_acc_proj_sub_label))

    vid_clusters_name = ['dance_1', 'acrob_1', 'magic_1', 'dance_2',
                         'mime_2', 'acrob_2', 'magic_2', 'dance_3',
                         'magic_3', 'dance_4', 'mime_3', 'dance_5',
                         'mime_6'] 
    # vid_clusters_name = ['2:lostthai', '1:worldhtchina', '0:a-shock', '0:bkto1942',
    #                      '1:worldhtchina', '2:lostthai', '0:bkto1942', '1:worldhtchina',
    #                      '2:flirtscho', '2:pandorasbox', '1:worldhtchina', '0:bkto1942',
    #                      '1:worldhtchina', '2:pandorasbox', '0:a-shock'] 
    
    os.makedirs(args.save_path, exist_ok=True)
    # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    # tsne_plot.plot(eeg_feat_vec, eeg_feat_vid_label, clustering_acc_proj_vid_label, 'vid_label', 'vid_label', args.save_path, file_name='train_vid_test_vid', data_type='test', cluster_names=vid_clusters_name)

    # emot_cluster_name = ['sad', 'neutral', 'happy']
    # tsne_plot = TsnePlot(perplexity=30, learning_rate=700, n_iter=1000)
    # tsne_plot.plot(eeg_feat_vec, eeg_feat_emot_label, clustering_acc_proj_emot_label, 'vid_label', 'emot_label', args.save_path, file_name='train_vid_test_emot', data_type='test', cluster_names=emot_cluster_name)

    sub_clusters_name = ['sub:0', 'sub:1', 'sub:2', 'sub:3', 'sub:4', 'sub:5',
                         'sub:6', 'sub:7', 'sub:8', 'sub:9', 'sub:10', 'sub:11',
                         'sub:12', 'sub:13', 'sub:14', 'sub:15', 'sub:16', 'sub:17',
                         'sub:18', 'sub:19',] 

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
    tsne_results = tsne.fit_transform(eeg_feat_vec)
    df = pd.DataFrame()
    file_name = 'train_vid_test_vid'

    df['label'] = eeg_feat_vid_label
    df['x1'] = tsne_results[:, 0]
    df['x2'] = tsne_results[:, 1]
    df.to_csv(args.save_path + f'/{file_name}.csv')	

    df = pd.read_csv(args.save_path + f'/{file_name}.csv')

    plt.figure(figsize=(12,10))

    # ax = plt.axes(projection='3d')
    sns.scatterplot(
        x="x1", y="x2",
        data=df,
        hue='label',
        palette=sns.color_palette("hls", args.vid_labels),
        legend="full",
        alpha=0.4
    )

    cluster_centers = df.groupby("label")[["x1", "x2"]].mean()

    # Annotate clusters with names
    for label, (center_x, center_y) in cluster_centers.iterrows():
        cluster_name = vid_clusters_name[int(label)]  # Assign correct cluster name
        plt.text(center_x, center_y, cluster_name, fontsize=12, fontweight="bold",
                ha="center", va="center", bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    plt.savefig(args.save_path + f'/{file_name}.pdf', bbox_inches='tight', dpi=300)



    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
    tsne_results = tsne.fit_transform(eeg_feat_vec)
    df = pd.DataFrame()
    file_name = 'train_vid_test_sub'

    df['label'] = eeg_feat_sub_label
    df['x1'] = tsne_results[:, 0]
    df['x2'] = tsne_results[:, 1]
    df.to_csv(args.save_path + f'/{file_name}.csv')	

    df = pd.read_csv(args.save_path + f'/{file_name}.csv')

    plt.figure(figsize=(12,10))

    # ax = plt.axes(projection='3d')
    sns.scatterplot(
        x="x1", y="x2",
        data=df,
        hue='label',
        palette=sns.color_palette("hls", args.sub_labels),
        legend="full",
        alpha=0.4
    )

    cluster_centers = df.groupby("label")[["x1", "x2"]].mean()

    # Annotate clusters with names
    for label, (center_x, center_y) in cluster_centers.iterrows():
        cluster_name = sub_clusters_name[int(label)]  # Assign correct cluster name
        plt.text(center_x, center_y, cluster_name, fontsize=12, fontweight="bold",
                ha="center", va="center", bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    plt.savefig(args.save_path + f'/{file_name}.pdf', bbox_inches='tight', dpi=300)