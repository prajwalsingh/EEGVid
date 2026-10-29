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
import umap.umap_ as umap
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import re

style.use('ggplot')

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

    running_loss         = []
    eeg_featvec_proj     = np.array([])
    labels_array         = np.array([])
    subject_array         = np.array([])

    Enc_eeg.eval()
    # Proj_eeg.eval()

    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, labels, subject_id) in enumerate(tq, start=1):
        eeg         = eeg.to(args.device).unsqueeze(dim=1)
        labels      = labels.to(args.device)
        subject_id  = subject_id.to(args.device)

        with torch.no_grad():
            x_proj = Enc_eeg(eeg)

        eeg_featvec_proj     = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array         = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()
        subject_array        = np.concatenate((subject_array, subject_id.cpu().detach().numpy()), axis=0) if subject_array.size else subject_id.cpu().detach().numpy()

    return eeg_featvec_proj, labels_array, subject_array


def parse_args():
    parser = argparse.ArgumentParser('argument for EEG feature extraction')

    parser.add_argument('--eeg_path', type=str, default=None, help='downstream dataset name')
    parser.add_argument('--data_aug', nargs='*', default=None, help="['gauss', 'timerev','timemask']")
    parser.add_argument('--exp_type', type=str, required=True, help='which EEG channel to use ["all", "left", "right"]')
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
    
    dir_info  = natsorted(glob('EXPERIMENT_EEG2FeatSEED_DV_{}_*/*'.format(args.exp_type)))
    
    for experiment_path in tqdm(dir_info):
        subject_number = int(experiment_path.split(os.path.sep)[-1].split('_')[-1])
        subject_id = ''.join(experiment_path.split(os.path.sep)[-1].split('_'))
        eeg_path = args.eeg_path + subject_id + '.npy'

        k_fold_avg_kmean_acc  = 0.0
        k_fold_avg_linear_acc = 0.0

        for k_fold_id in range(0, 7):

            print(f'Running K-Fold: {k_fold_id+1}/7')

            ckpt_path = experiment_path+'/'+f'kfold_{k_fold_id}'+'/bestckpt/eegfeat_all.pth'

            complete_eeg_data_train    = torch.tensor([])
            complete_eeg_label_train   = torch.tensor([])
            complete_eeg_subject_train = torch.tensor([])

            complete_eeg_data_val    = torch.tensor([])
            complete_eeg_label_val   = torch.tensor([])
            complete_eeg_subject_val = torch.tensor([])

            complete_eeg_data_test    = torch.tensor([])
            complete_eeg_label_test   = torch.tensor([])
            complete_eeg_subject_test = torch.tensor([])

            eegdata = np.load(eeg_path)

            # print(eegdata.shape)

            chosed_index = []
            for i in range(7):
                index = [list(GT_label[i]).index(element) for element in chosed_label]
                chosed_index.append(index)
            new_eeg = np.zeros((7, 40, 5, 62, 400))
            new_eeg_labels = []
            for i in range(7):
                new_eeg[i] = eegdata[i][chosed_index[i], :, :, :]
                new_eeg_labels.append(GT_label[i][chosed_index[i]])

            new_eeg = torch.from_numpy(new_eeg)
            new_eeg_labels = torch.from_numpy(np.array(new_eeg_labels))

            new_eeg_labels = new_eeg_labels.unsqueeze(-1).expand(-1, -1, 5)

            test_idx = ((k_fold_id+1)%7)
            test_eeg = new_eeg[test_idx:test_idx+1,...]
            test_eeg_labels = new_eeg_labels[test_idx:test_idx+1, ...]

            ## As per paper, val block is always one behind the test
            val_idx = ((k_fold_id)%7)
            val_eeg = new_eeg[val_idx:val_idx+1,...]
            val_eeg_labels = new_eeg_labels[val_idx:val_idx+1, ...]
            
            train_idxs = list(range(0, 7))
            train_idxs.remove(test_idx)
            train_idxs.remove(val_idx)
            train_eeg  = new_eeg[train_idxs, ...]
            train_eeg_labels = new_eeg_labels[train_idxs , ...]

            train_eeg        = torch.reshape(train_eeg, (train_eeg.shape[0] * train_eeg.shape[1] * train_eeg.shape[2], train_eeg.shape[3], train_eeg.shape[4]))
            train_eeg_labels = torch.reshape(train_eeg_labels, (train_eeg_labels.shape[0] * train_eeg_labels.shape[1] * train_eeg_labels.shape[2],))

            val_eeg        = torch.reshape(val_eeg,(val_eeg.shape[0] * val_eeg.shape[1] * val_eeg.shape[2], val_eeg.shape[3], val_eeg.shape[4]))
            val_eeg_labels = torch.reshape(val_eeg_labels, (val_eeg_labels.shape[0] * val_eeg_labels.shape[1] * val_eeg_labels.shape[2],))

            test_eeg        = torch.reshape(test_eeg,(test_eeg.shape[0] * test_eeg.shape[1] * test_eeg.shape[2], test_eeg.shape[3], test_eeg.shape[4]))
            test_eeg_labels = torch.reshape(test_eeg_labels, (test_eeg_labels.shape[0] * test_eeg_labels.shape[1] * test_eeg_labels.shape[2],))

            subject_labels_train =  torch.from_numpy(np.array([subject_number]*train_eeg.shape[0]))
            subject_labels_val   =  torch.from_numpy(np.array([subject_number]*val_eeg.shape[0]))
            subject_labels_test  =  torch.from_numpy(np.array([subject_number]*test_eeg.shape[0]))

            complete_eeg_data_train    = torch.cat([complete_eeg_data_train, train_eeg], dim=0)
            complete_eeg_label_train   = torch.cat([complete_eeg_label_train, train_eeg_labels], dim=0)
            complete_eeg_subject_train = torch.cat([complete_eeg_subject_train, subject_labels_train], dim=0)

            complete_eeg_data_val    = torch.cat([complete_eeg_data_val, val_eeg], dim=0)
            complete_eeg_label_val   = torch.cat([complete_eeg_label_val, val_eeg_labels], dim=0)
            complete_eeg_subject_val = torch.cat([complete_eeg_subject_val, subject_labels_val], dim=0)

            complete_eeg_data_test    = torch.cat([complete_eeg_data_test, test_eeg], dim=0)
            complete_eeg_label_test   = torch.cat([complete_eeg_label_test, test_eeg_labels], dim=0)
            complete_eeg_subject_test = torch.cat([complete_eeg_subject_test, subject_labels_test], dim=0)

            # print(new_eeg.shape, new_eeg_labels.shape, test_eeg.shape, test_eeg_labels.shape)
            # print(complete_eeg_data_train.shape, complete_eeg_label_train.shape, complete_eeg_subject_train.shape)

        # total_unique_classes_train = torch.unique(complete_eeg_label_train)
        # total_unique_classes_test = torch.unique(complete_eeg_label_test)

        # total_unique_subject_train = torch.unique(complete_eeg_subject_train)
        # total_unique_subject_test = torch.unique(complete_eeg_subject_test)

            print(complete_eeg_data_train.shape, complete_eeg_label_train.shape, complete_eeg_subject_train.shape)
            print(complete_eeg_data_val.shape, complete_eeg_label_val.shape, complete_eeg_subject_val.shape)
            print(complete_eeg_data_test.shape, complete_eeg_label_test.shape, complete_eeg_subject_test.shape)
            # print('Total video classes train:')
            # print(total_unique_classes_train)
            # print('Total video classes test:')
            # print(total_unique_classes_test)
            # print('Total subjects train:')
            # print(total_unique_subject_train)
            # print('Total subjects test:')
            # print(total_unique_subject_test)

            train_dataset  = VideoEEGDataset(complete_eeg_data_train, complete_eeg_label_train, complete_eeg_subject_train, args=args)
            val_dataset    = VideoEEGDataset(complete_eeg_data_val, complete_eeg_label_val, complete_eeg_subject_val, args=None)
            test_dataset   = VideoEEGDataset(complete_eeg_data_test, complete_eeg_label_test, complete_eeg_subject_test, args=None)


            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
            val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
            test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

            # print('*'*50)
            # print(f'Experiment type: {args.exp_type}, Total channels: {EEG_CAP_SHOT_V1[args.exp_type].shape[0]}')
            # print('*'*50)

            # Enc_eeg  = gat_network.EEGEncoder(eeg_channels=EEG_CAP_SHOT_V1[args.exp_type].shape[0], in_timesteps=args.eeg_timesteps, exp_type=args.exp_type).to(args.device)
            
            Enc_eeg  = gat_network.EEGEncoder(eeg_channels=62, in_timesteps=args.eeg_timesteps, exp_type=args.exp_type).to(args.device)
            Enc_eeg.apply(gat_network.weights_init_normal)

            ######################################################
            print(f'Using checkpoint: {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            Enc_eeg.load_state_dict(checkpoint['enc_eeg_state_dict'])
            START_EPOCH = checkpoint['epoch']
            print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
            ######################################################
            
            eeg_feat_vec_train, eeg_feat_label_train, eeg_feat_sub_label_train  = validation(START_EPOCH, Enc_eeg, train_dataloader, args)
            eeg_feat_vec_test, eeg_feat_label_test, eeg_feat_sub_label_test  = validation(START_EPOCH, Enc_eeg, test_dataloader, args)

            ### compute k-means score and Umap score on the text and image embeddings
            k_means             = K_means(n_clusters=args.num_classes)
            clustering_acc_proj = k_means.transform(eeg_feat_vec_test, eeg_feat_label_test)
            k_fold_avg_kmean_acc += clustering_acc_proj
            # print('\nTest KMeans clustering score kfold_{} of subject {}: {}'.format(k_fold_id, subject_id, clustering_acc_proj))
            #########################################################################
            
            # Source: https://www.baeldung.com/cs/svm-multiclass-classification

            poly_lin = svm.LinearSVC(multi_class='ovr', verbose=1, random_state=45, C=1.0, max_iter=1000).fit(eeg_feat_vec_train, eeg_feat_label_train)
            # rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1, verbose=1, random_state=45, max_iter=1000).fit(eeg_feat_vec_train, eeg_feat_label_train)
            # poly = svm.SVC(kernel='poly', degree=3, C=1, verbose=1, random_state=45, max_iter=1000).fit(eeg_feat_vec_train, eeg_feat_label_train)


            poly_lin_pred = poly_lin.predict(eeg_feat_vec_test)
            # poly_pred = poly.predict(eeg_feat_vec_test)
            # rbf_pred = rbf.predict(eeg_feat_vec_test)

            poly_lin_accuracy = accuracy_score(eeg_feat_label_test, poly_lin_pred)
            poly_lin_f1 = f1_score(eeg_feat_label_test, poly_lin_pred, average='weighted')
            k_fold_avg_linear_acc += poly_lin_accuracy
            # print('Accuracy (Linear): ', "%.2f" % (poly_lin_accuracy*100))
            # print('F1 (Linear): ', "%.2f" % (poly_lin_f1*100))

            # poly_accuracy = accuracy_score(eeg_feat_label_test, poly_pred)
            # poly_f1 = f1_score(eeg_feat_label_test, poly_pred, average='weighted')
            # print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
            # print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

            # rbf_accuracy = accuracy_score(eeg_feat_label_test, rbf_pred)
            # rbf_f1 = f1_score(eeg_feat_label_test, rbf_pred, average='weighted')
            # print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
            # print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

        print('\n')
        print('='*50)
        print(f'[subject: {subject_id}]')    
        print(f'Average k-fold test k-means acc: {k_fold_avg_kmean_acc/7.0}')
        print(f'Average k-fold test linear acc: {k_fold_avg_linear_acc/7.0}')
        print('='*50)
        print('\n')

        with open('test_log.txt', 'a') as file:
            file.write('\n')
            file.write('='*50)
            file.write(f'\n[subject: {subject_id}]')    
            file.write(f'\nAverage k-fold test k-means acc: {k_fold_avg_kmean_acc/7.0}')
            file.write(f'\nAverage k-fold test linear acc: {k_fold_avg_linear_acc/7.0}')
            file.write('\n')
            file.write('='*50)