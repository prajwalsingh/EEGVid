import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import shutil

def save_video(video, path):
    height, width = 128, 128
    fps = 8
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    # Convert filtered frames to NumPy and write to video
    for frame_idx in range(fps):
        frame = video[:, (frame_idx*128):(frame_idx*128)+128, :]
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Filtered frames saved as video at: {path}")

if __name__ == '__main__':

    generated_video_paths = natsorted(glob('generated_videos_002_v2_ckpt3760/*'))

    os.makedirs('generated_clip_gt', exist_ok=True)
    os.makedirs('generated_clip_gen', exist_ok=True)

    for path in tqdm(generated_video_paths):
        filename = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path, 1)
        gt_image = image[:128, :, :]
        pred_image = image[128:, :, :]

        gt_path = os.path.join('generated_clip_gt', f'{filename}.mp4')
        gen_path = os.path.join('generated_clip_gen', f'{filename}.mp4')

        save_video(gt_image, gt_path)
        save_video(pred_image, gen_path)

