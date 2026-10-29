#%%
import os
import re
import mne
import glob
import numpy as np
from tqdm.auto import tqdm
# from moviepy.editor import VideoFileClip
from mne.preprocessing import EOGRegression
# %%

class_map = {
    "01_eeg": 0, "03_eeg": 1, "04_eeg": 2, "05_eeg": 3, "06_eeg": 4, "07_eeg": 5, "08_eeg": 6, "09_eeg": 7, "12_eeg": 8, "13_eeg": 9, "14_eeg": 10, "15_eeg": 11, "16_eeg": 12
}

channel_map = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
    'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7',
    'A16': 'TP7', 'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5',
    'A23': 'P7', 'A24': 'P9', 'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz',
    'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz', 'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4',
    'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4', 'B9': 'F6', 'B10': 'F8', 'B11': 'FT8',
    'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz', 'B17': 'C2', 'B18': 'C4',
    'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2', 'B25': 'P2',
    'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'
}

## paths are os dependent, change for windows
path_to_dataset = "" ## Set the path to dataset here
datadir = f"{path_to_dataset}/dataset/ACTION/SingleShot"
notch = (50,)       # Notch filter 50 Hz or None
filter_l = 0.5        # High pass cutoff
filter_h = None       # Low pass cutoff
segment_length = 2  # 2 second segment/epoch
overlap = 0         # 0 second overlap
downsample = 200   # Downsampling frequency
output_dir = f"{path_to_dataset}/dataset/ACTION/Preprocessed_SingleShot/segment_notch_{re.sub(r'[(,)]','',str(notch))}Hz_bandpassfilter_{str(filter_l).replace('.', '_')}_{str(filter_h).replace('.', '_')}Hz_len_{str(segment_length).replace('.', '_')}s_overlap_{str(overlap).replace('.', '_')}s_downsampling_{str(downsample)}_1"
montage = mne.channels.make_standard_montage('biosemi64')
os.makedirs(output_dir, exist_ok=False)
# %%
# Chunking EEG
# ============


for sub_dir in sorted(os.listdir(datadir)):
    sub_path = os.path.join(datadir, sub_dir)
    sub_num = int(sub_dir.split("_")[1]) - 1
    source_eeg_paths = sorted(glob.glob(os.path.join(datadir, sub_dir, "*.set")))
    
    # Load bad channel information
    bad_channels_file = os.path.join(sub_path, "README.txt")
    bad_channels_list = []

    with open(bad_channels_file, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                _, bad_chs = parts
                bad_channels_list.extend([channel_map[ch.strip()] for ch in bad_chs.split(',') if ch.strip() != "None"])
    
    for source_path in source_eeg_paths:
        print("=====================================================================================================")
        print("Processing:", source_path)
        print("=====================================================================================================")

        raw = mne.io.read_raw_eeglab(source_path, preload=True, eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'])
        raw = raw.rename_channels(mapping=channel_map)
        raw = raw.set_montage(montage)

        # events = mne.events_from_annotations(raw)[0]
        # if(events.shape[0]>1):
        #     with open("error_log.txt", "a") as f:
        #         f.write(f"Boundary events {source_path}: {events}\n")
        #     break

        if len(bad_channels_list) > 0:
            raw.info['bads'] = bad_channels_list

        try:        
            ## Pre-processing

            ### 1. Interpolating bad channels
            if len(raw.info['bads']) > 0:
                print(f"Interpolating bad channels: {raw.info['bads']}")
                raw = raw.interpolate_bads()

            ### 2. Re-referencing to average
            raw = raw.set_eeg_reference(ref_channels='average')

            ### 3. Apply notch filter to remove power line noise
            raw = raw.notch_filter(notch)

            ### 4. Apply high-pass filter
            raw = raw.filter(l_freq=filter_l, h_freq=filter_h)

            ### 5. Downsampling
            raw = raw.resample(downsample)

            ### 6. Regressing out the EOG channels to reduce eye artifacts
            eog_regression = EOGRegression(picks='eeg', picks_artifact='eog', proj=False).fit(raw)
            raw_clean = eog_regression.apply(raw)

            ### 7. Epoching
            epochs = mne.make_fixed_length_epochs(raw_clean, duration=segment_length, preload=True, overlap=overlap, proj=False)

            eeg_data = epochs.get_data(picks=['eeg'])
            class_name = os.path.basename(source_path).replace(".set", "")
            class_num = class_map[class_name]
            name = f"{sub_num}_{class_num}"

            rng = np.random.default_rng(seed=42)
            idx = np.arange(eeg_data.shape[0])
            rng.shuffle(idx)
            train_end = int(0.8*len(idx))
            val_end = train_end + ((len(idx) - train_end) // 2)

            ## train split
            train_path = os.path.join(output_dir, "eeg", "train")
            os.makedirs(train_path, exist_ok=True)
            for i in tqdm(idx[0:train_end], desc="Storing Train EEG epochs", leave=False):
                eeg_name = f"{name}_{idx[i]}.npy"
                np.save(os.path.join(train_path, eeg_name), eeg_data[idx[i]])
            
            ## val split
            val_path = os.path.join(output_dir, "eeg", "val")
            os.makedirs(val_path, exist_ok=True)
            for i in tqdm(idx[train_end:val_end], desc="Storing Val EEG epochs", leave=False):
                eeg_name = f"{name}_{idx[i]}.npy"
                np.save(os.path.join(val_path, eeg_name), eeg_data[idx[i]])
            
            ## test split
            test_path = os.path.join(output_dir, "eeg", "test")
            os.makedirs(test_path, exist_ok=True)
            for i in tqdm(idx[val_end:], desc="Storing Test EEG epochs", leave=False):
                eeg_name = f"{name}_{idx[i]}.npy"
                np.save(os.path.join(test_path, eeg_name), eeg_data[idx[i]])
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"Error processing {source_path}: {e}\n")


# %%
