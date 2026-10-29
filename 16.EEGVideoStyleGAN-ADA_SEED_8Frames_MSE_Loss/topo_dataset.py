#%%
import os
import re
import mne
import glob
import torch
import pickle
import models
import numpy as np
import multiprocessing
from tqdm.auto import tqdm
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision.io import read_image
from interpolator import _plot_topomap

# %%
class xDataset(Dataset):

    def __init__(self, sub, split, model, datadir, device):
        self.model = model
        self.device = device
        self.split = split
        self.sub = sub
        self.datadir = datadir

        assert model in ["topo", "eeg_conv", "eeg_gat"]

        ## Fetch image paths
        img_path  = os.path.join(datadir, "image_set")
        if split == "train":
            self.img_paths = sorted(glob.glob(f"{img_path}/training_images/**/*.jpg", recursive=True))
        else:
            self.img_paths = sorted(glob.glob(f"{img_path}/test_images/**/*.jpg", recursive=True))

        ## initialize checkpoints
        if model == "topo":
            self.Enc_eeg  = models.Enc_eeg_topo().to(device).eval()
            self.Proj_eeg = models.Proj_eeg(drop_proj=0.3).to(device).eval()
            self.Enc_eeg.load_state_dict(torch.load(os.path.join(datadir, "checkpoints",  "topo", f"sub{format(sub, '02')}_test512Enc_eeg_cls.pth")), strict=True)
            self.Proj_eeg.load_state_dict(torch.load(os.path.join(datadir, "checkpoints", "topo", f"sub{format(sub, '02')}_test512Proj_eeg_cls.pth")), strict=True)
        elif model == "eeg_gat":
            self.Enc_eeg  = models.Enc_eeg_gat().to(device).eval()
            self.Proj_eeg = models.Proj_eeg().to(device).eval()
            self.Enc_eeg.load_state_dict(torch.load(os.path.join(datadir, "checkpoints",  "eeg_gat", f"sub{format(sub, '02')}_test_gat_512Enc_eeg_cls.pth")), strict=True)
            self.Proj_eeg.load_state_dict(torch.load(os.path.join(datadir, "checkpoints", "eeg_gat", f"sub{format(sub, '02')}_test_gat_512Proj_eeg_cls.pth")), strict=True)
        else:
            self.Enc_eeg  = models.Enc_eeg().to(device).eval()
            self.Proj_eeg = models.Proj_eeg().to(device).eval()
            self.Enc_eeg.load_state_dict(torch.load(os.path.join(datadir, "checkpoints",  "eeg_conv", f"sub{format(sub, '02')}_test_conv_512Enc_eeg_cls.pth")), strict=True)
            self.Proj_eeg.load_state_dict(torch.load(os.path.join(datadir, "checkpoints", "eeg_conv", f"sub{format(sub, '02')}_test_conv_512Proj_eeg_cls.pth")), strict=True)
        
        ## load eeg data
        if model == "topo":
            ## Get EEG paths
            sub_path  = os.path.join(datadir, "Preprocessed_topomaps_250Hz", f"sub-{format(sub, '02')}", split)
            self.eeg_paths = sorted(glob.glob(f"{sub_path}/*.npy"))
        else:
            split = "training" if split == "train" else "test"
            data = np.load(os.path.join(datadir, "Preprocessed_data_250Hz", 'sub-' + format(sub, '02'), f'preprocessed_eeg_{split}.npy'), allow_pickle=True)
            data = data['preprocessed_eeg_data']
            data = np.mean(data, axis=1).astype(np.float32)
            data = np.expand_dims(data, axis=1) # for channel in conv
            self.eegs = torch.from_numpy(np.expand_dims(data, axis=1)) # for batchsize of one to get embedding in the fly
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index) -> tuple:
        if self.model == "topo":
            eeg = np.expand_dims(np.load(self.eeg_paths[index]), axis=0).astype(np.float32)
            eeg = torch.from_numpy(eeg).to(self.device)
        else:
            eeg = self.eegs[index].to(self.device)
        eeg_emb = self.Proj_eeg(self.Enc_eeg(eeg))

        img = read_image(self.img_paths[index])
        class_name = re.search("/\d{5}_.+/", self.img_paths[index]).group(0)[7:-1]
        if self.split == "train":
            return torch.squeeze(eeg_emb), eeg, img, index // 10, class_name
        else:
            return torch.squeeze(eeg_emb), eeg, img, index, class_name


def scale(low,high,data):
    mi = data.min()
    ma = data.max()
    data = (high-low)*((data-mi)/(ma-mi)) + low
    return data

def func(eeg_t,info,res):
    return torch.from_numpy(scale(-1, 1, _plot_topomap(eeg_t,info,outlines=None,res=res)))

class xDropTest(Dataset):

    def __init__(self, input_set: Subset, ch_idx_to_drop: list):
        self.input_set = input_set
        ch_idx_to_drop = [i-1 for i in ch_idx_to_drop] if ch_idx_to_drop else None

        ## initialize checkpoints
        self.Enc_eeg  = input_set.dataset.Enc_eeg
        self.Proj_eeg = input_set.dataset.Proj_eeg

        ## load eeg data
        if input_set.dataset.model == "topo":
            ## Get EEG paths
            sub_path = os.path.join(self.input_set.dataset.datadir, "Preprocessed_data_250Hz", f"sub-{format(self.input_set.dataset.sub, '02')}", "preprocessed_eeg_training.npy")
            self.eegs = np.load(sub_path, allow_pickle=True)['preprocessed_eeg_data']
            self.eegs = np.mean(self.eegs, axis=1)[self.input_set.indices]

            ## Get montage
            montage_path = os.path.join(self.input_set.dataset.datadir, "montage.pickle")
            temp = open(montage_path, "rb")
            reference_montage = pickle.load(temp)
            temp.close()
            reference_ch_names = reference_montage.ch_names
            reference_ch_types = ["eeg"] * len(reference_ch_names)
            info = mne.create_info(ch_names=reference_ch_names, ch_types=reference_ch_types, sfreq=250)
            info = info.set_montage(reference_montage)

            ##Get montage with dropped channel
            num_ch_drop = len(ch_idx_to_drop)
            ch_to_drop = list(np.array(reference_ch_names)[ch_idx_to_drop])
            print(f"==============================")
            print(f"Channels to drop: {ch_to_drop}")
            print(f"==============================")
            raw = mne.io.RawArray(self.eegs[0], info)
            raw.drop_channels(ch_to_drop)
            assert raw.get_data().shape[0] == len(reference_ch_names)-num_ch_drop and len(raw.info.ch_names) == len(reference_ch_names)-num_ch_drop, "Error in subsampling"
            dropped_info = raw.info

            ##Get index of of channels to be considered
            subset_index = []
            for ch in reference_ch_names:
                if ch not in ch_to_drop:
                    subset_index.append(reference_ch_names.index(ch))
            assert len(subset_index) == len(reference_ch_names)-num_ch_drop, "Error in subsampling"
            pool = multiprocessing.Pool(8)
            topos = []
            for sample in tqdm(self.eegs, desc="Generating randomly dropped topo maps", leave=False):
                sample = sample[subset_index,:]
                processes = [pool.apply_async(func, args=(sample[:,i], dropped_info, 32)) for i in range(self.eegs.shape[2])]
                topos.append(torch.stack([p.get() for p in processes]).type(torch.float).unsqueeze(dim=0))
            self.eegs = topos
        else:
            data = np.load(os.path.join(self.input_set.dataset.datadir, "Preprocessed_data_250Hz", 'sub-' + format(self.input_set.dataset.sub, '02'), 'preprocessed_eeg_training.npy'), allow_pickle=True)
            data = data['preprocessed_eeg_data']
            data[:,:,ch_idx_to_drop,:] = 0.0
            data = np.mean(data, axis=1).astype(np.float32)[self.input_set.indices]
            data = np.expand_dims(data, axis=1) # for channel in conv
            self.eegs = torch.from_numpy(np.expand_dims(data, axis=1)) # for batchsize of one to get embedding in the fly
        
        assert len(self.eegs) == len(self.input_set)
    
    def __len__(self):
        return len(self.input_set)
    
    def __getitem__(self, index) -> tuple:
        eeg = self.eegs[index].to(self.input_set.dataset.device)
        eeg_emb = self.Proj_eeg(self.Enc_eeg(eeg))
        
        img = self.input_set[index][2]

        class_id = self.input_set[index][3]
        class_name = self.input_set[index][4]
        return torch.squeeze(eeg_emb), eeg, img, class_id, class_name

