import os
from glob import glob
from natsort import natsorted

if __name__ == '__main__':
    
    paths = natsorted(glob('filter_clips_v2_ckpt3760/*.mp4'))
    
    save_dir = 'filter_clips_v2_ckpt3760_gif'
    os.makedirs(save_dir, exist_ok=True)
    
    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        os.system(f'ffmpeg -i {path} {save_dir+"/" + filename + ".gif"}')