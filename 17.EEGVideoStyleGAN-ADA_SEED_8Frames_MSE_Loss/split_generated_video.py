import os
import cv2
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import shutil

if __name__ == '__main__':

    generated_video_paths = natsorted(glob('generated_videos_002_v2_ckpt3760/*'))

    os.makedirs('generated_videos_gt', exist_ok=True)
    os.makedirs('generated_videos_gen', exist_ok=True)

    for path in tqdm(generated_video_paths):
        filename = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path, 1)
        gt_image = image[:128, :, :]
        pred_image = image[128:, :, :]
        cv2.imwrite(os.path.join('generated_videos_gt', f'{filename}.png'), gt_image)
        cv2.imwrite(os.path.join('generated_videos_gen', f'{filename}.png'), pred_image)


