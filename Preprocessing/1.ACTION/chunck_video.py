#%%
import os
import re
import mne
import glob
import numpy as np
from tqdm.auto import tqdm
from moviepy.editor import VideoFileClip
from mne.preprocessing import EOGRegression
# %%

class_map = {
    "01_eeg": 0, "03_eeg": 1, "04_eeg": 2, "05_eeg": 3, "06_eeg": 4, "07_eeg": 5, "08_eeg": 6, "09_eeg": 7, "12_eeg": 8, "13_eeg": 9, "14_eeg": 10, "15_eeg": 11, "16_eeg": 12
}

video_class_map = {
    "01_Dance_1" : 0,  
    "03_Acrob_1" : 1,  
    "04_Magic_1" : 2,  
    "05_Dance_2" : 3,  
    "06_Mime_2"  : 4,  
    "07_Acrob_2" : 5,  
    "08_Magic_2" : 6,  
    "09_Dance_3" : 7,  
    "12_Magic_3" : 8,  
    "13_Dance_4" : 9,  
    "14_Mime_3"  : 10, 
    "15_Dance_5" : 11, 
    "16_Mime_6"  : 12, 
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
output_dir = f"{path_to_dataset}/dataset/ACTION/Preprocessed_SingleShot/segment_notch_{re.sub(r'[(,)]','',str(notch))}Hz_bandpassfilter_{str(filter_l).replace('.', '_')}_{str(filter_h).replace('.', '_')}Hz_len_{str(segment_length).replace('.', '_')}s_overlap_{str(overlap).replace('.', '_')}s_downsampling_{str(downsample)}"
chunk_path = os.path.join(output_dir, "video")
os.makedirs(chunk_path, exist_ok=False)

# %%
# Chunking Video
# ==============
source_video_dir = f"{path_to_dataset}/dataset/ACTION/SingleShotVideo"
source_video_paths = sorted(glob.glob(os.path.join(source_video_dir, "*.mp4")))
for source_path in source_video_paths:
    print("=====================================================================================================")
    print("Processing:", source_path)
    print("=====================================================================================================")
    clip = VideoFileClip(source_path)
    duration = clip.duration
    chunk_duration = segment_length
    shift = chunk_duration - overlap
    
    # Calculate number of chunks
    # num_chunks = int(duration // chunk_duration)
    num_chunks = int((duration - chunk_duration + shift) // shift)
    class_name = os.path.basename(source_path).replace(".mp4", "")
    class_id = video_class_map[class_name]

    for chunk_id in range(num_chunks):
        start_time = chunk_id * shift
        end_time = start_time + chunk_duration
        
        # Create a subclip
        subclip = clip.subclip(start_time, end_time)
        file_name = f"{class_id}_{chunk_id}.mp4"
        subclip.write_videofile(os.path.join(chunk_path, file_name), codec="libx264")
        print(f"Created chunk: {file_name}")
# %%
