# %%
# Arguments
# =========
window_size     = 400 #(correspondin to 2s)
shift           = 400 #(correspondin to 2s) 
# %%
# Imports
# =======
from scipy.io import loadmat
from tqdm.auto import tqdm
import numpy as np
import os
# %%
# Constants
# =========
PATH_TO_DATASET                 = "" ## Set the path to dataset here
SOURCE                          = f"{PATH_TO_DATASET}/dataset/SEED/Preprocessed_EEG/data"
TARGET                          = f"{PATH_TO_DATASET}/dataset/SEED/eeg_video_dataset_random_split_v2_tokens/"
VIDEO_CHUNKS                    = f"{PATH_TO_DATASET}/dataset/SEED/video_chunks/"
VIDEO_CHUNKS_CAPTION            = f"{PATH_TO_DATASET}/dataset/SEED/video_chunks_caption/"
VIDEO_CHUNKS_CAPTION_TOKENS     = f"{PATH_TO_DATASET}/dataset/SEED/video_chunks_tokens/"
## as per dataset, all trials are recored in the following order of emotion
LABELS = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
LABELS_MAP = {-1:0, 0:1, 1:2}
# %%
# Processing
# ==========
## sorting below will make each subject's experiments together
all_exp_paths = [os.path.join(SOURCE, i) for i in sorted(os.listdir(SOURCE))]
all_exp_paths = np.array(all_exp_paths).reshape(-1,3)

for subject_exps in all_exp_paths:
    for repeat, subject_exp_path in enumerate(subject_exps):
        subject_num = subject_exp_path[subject_exp_path.rfind("/")+1:subject_exp_path.rfind("_")]
        repeat_path = os.path.join(TARGET, f"repeat_{repeat}")
        os.makedirs(repeat_path, exist_ok=True)
        exp = loadmat(subject_exp_path)

        for trial_id, trial_name in enumerate(list(exp.keys())[3:]):
            eeg = exp[trial_name]

            ## Creating train, val, test split indices
            total_samples = (eeg.shape[1] - window_size + shift) // shift
            total_samples = 120 if total_samples > 120 else total_samples
            all_indices = np.arange(total_samples)
            ## Fixing seed as we need same arrangement always.
            ## Otherwise there would be info leak. 
            rng = np.random.default_rng(seed=42)
            rng.shuffle(all_indices) 
            train_end = int(total_samples*0.8)
            val_start = train_end
            val_end = val_start + (total_samples-train_end)//2
            test_start = val_end

            ## Save train split
            train_path = os.path.join(repeat_path, "train")
            os.makedirs(train_path, exist_ok=True)

            for idx in tqdm(all_indices[:train_end], desc=f"Train_split | Subject: {subject_num} | Repeat: {repeat} | Trial_{trial_id+1}: {trial_name}"):
                if idx*shift+window_size <= eeg.shape[1]:
                    sample = dict()
                    sample['trial_id'] = trial_id+1
                    sample['emotion']  = LABELS_MAP[LABELS[trial_id]]
                    video = f"{sample['trial_id']}_{sample['emotion']}_{idx}"

                    ## Get video caption
                    file = open(os.path.join(VIDEO_CHUNKS_CAPTION, f"{video}.txt"), "r")
                    video_text = file.read()
                    file.close()
                    sample['video_text'] = video_text

                    ## Get video caption embedding
                    video_text_token = np.load(os.path.join(VIDEO_CHUNKS_CAPTION_TOKENS, f"{video}.npy"), allow_pickle=True)
                    sample['video_text_token_with_mask'] = video_text_token

                    ## Get video file name
                    video_name = f"{video}.mp4"
                    ## Just checking if mapping is fine
                    assert os.path.exists(os.path.join(VIDEO_CHUNKS, video_name)), f"{video_name} does not exists"
                    sample['video_name'] = video_name

                    ## Get EEG chunk
                    sample['eeg'] = eeg[:, idx*shift:idx*shift+window_size]
                    np.save(os.path.join(train_path, f"{subject_num}_{video}.npy"), sample)
            
            ## Save val split
            val_path = os.path.join(repeat_path, "val")
            os.makedirs(val_path, exist_ok=True)

            for idx in tqdm(all_indices[val_start:val_end], desc=f"Val_split | Subject: {subject_num} | Repeat: {repeat} | Trial_{trial_id+1}: {trial_name}"):
                if idx*shift+window_size <= eeg.shape[1]:
                    sample = dict()
                    sample['trial_id'] = trial_id+1
                    sample['emotion']  = LABELS_MAP[LABELS[trial_id]]
                    video = f"{sample['trial_id']}_{sample['emotion']}_{idx}"

                    ## Get video caption
                    file = open(os.path.join(VIDEO_CHUNKS_CAPTION, f"{video}.txt"), "r")
                    video_text = file.read()
                    file.close()
                    sample['video_text'] = video_text

                    ## Get video caption embedding
                    video_text_token = np.load(os.path.join(VIDEO_CHUNKS_CAPTION_TOKENS, f"{video}.npy"), allow_pickle=True)
                    sample['video_text_token_with_mask'] = video_text_token

                    ## Get video file name
                    video_name = f"{video}.mp4"
                    ## Just checking if mapping is fine
                    assert os.path.exists(os.path.join(VIDEO_CHUNKS, video_name)), f"{video_name} does not exists"
                    sample['video_name'] = video_name

                    ## Get EEG chunk
                    sample['eeg'] = eeg[:, idx*shift:idx*shift+window_size]
                    np.save(os.path.join(val_path, f"{subject_num}_{video}.npy"), sample)
            
            ## Save test split
            test_path = os.path.join(repeat_path, "test")
            os.makedirs(test_path, exist_ok=True)

            for idx in tqdm(all_indices[test_start:], desc=f"Test_split | Subject: {subject_num} | Repeat: {repeat} | Trial_{trial_id+1}: {trial_name}"):
                if idx*shift+window_size <= eeg.shape[1]:
                    sample = dict()
                    sample['trial_id'] = trial_id+1
                    sample['emotion']  = LABELS_MAP[LABELS[trial_id]]
                    video = f"{sample['trial_id']}_{sample['emotion']}_{idx}"

                    ## Get video caption
                    file = open(os.path.join(VIDEO_CHUNKS_CAPTION, f"{video}.txt"), "r")
                    video_text = file.read()
                    file.close()
                    sample['video_text'] = video_text

                    ## Get video caption embedding
                    video_text_token = np.load(os.path.join(VIDEO_CHUNKS_CAPTION_TOKENS, f"{video}.npy"), allow_pickle=True)
                    sample['video_text_token_with_mask'] = video_text_token

                    ## Get video file name
                    video_name = f"{video}.mp4"
                    ## Just checking if mapping is fine
                    assert os.path.exists(os.path.join(VIDEO_CHUNKS, video_name)), f"{video_name} does not exists"
                    sample['video_name'] = video_name

                    ## Get EEG chunk
                    sample['eeg'] = eeg[:, idx*shift:idx*shift+window_size]
                    np.save(os.path.join(test_path, f"{subject_num}_{video}.npy"), sample)
                        

        
# %%
