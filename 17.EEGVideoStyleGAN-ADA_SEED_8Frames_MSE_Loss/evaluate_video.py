import os
import torch
import torchvision.transforms as transforms
import lpips
import numpy as np
import imageio.v3 as imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.linalg import sqrtm
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LPIPS model
lpips_model = lpips.LPIPS(net='vgg').to(device)

# Load I3D model for FVD
# i3d_model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1")

# Transformation for images
transform = transforms.ToTensor()

# Paths to folders
gt_folder = "generated_videos_gt"
gen_folder = "generated_videos_gen"

# Load images and preprocess
def load_and_split_image(image_path):
    """Loads an image and splits it into 4 frames."""
    image = imageio.imread(image_path)
    image = transform(image).permute(1, 2, 0)
    frames = torch.chunk(image, chunks=4, dim=1)
    return [frame.unsqueeze(0).permute(0, 3, 1, 2) for frame in frames]

def compute_psnr(gt_frames, gen_frames):

    psnr_scores = []
    max_pixel_value = 1.0  # Assuming normalized images in [0, 1]

    for gt, gen in zip(gt_frames, gen_frames):
        # Ensure tensors are on the same device and have the same shape
        gt = gt.to(device)
        gen = gen.to(device)
        
        # Compute MSE
        mse = mse_loss(gen, gt)
        
        # Avoid division by zero
        if mse.item() == 0:
            psnr = float('inf')
        else:
            psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        
        psnr_scores.append(psnr.item())
    
    return np.mean(psnr_scores)


# Compute SSIM
def compute_ssim(gt_frames, gen_frames):
    ssim_scores = []
    for gt, gen in zip(gt_frames, gen_frames):
        gt_np = gt.squeeze().cpu().numpy().transpose(1, 2, 0)
        gen_np = gen.squeeze().cpu().numpy().transpose(1, 2, 0)
        ssim_scores.append(ssim(gt_np, gen_np, multichannel=True, data_range=1.0, channel_axis=2))
    return np.mean(ssim_scores)

# Compute LPIPS
def compute_lpips(gt_frames, gen_frames):
    lpips_scores = []
    for gt, gen in zip(gt_frames, gen_frames):
        lpips_score = lpips_model(gt.to(device), gen.to(device))
        lpips_scores.append(lpips_score.item())
    return np.mean(lpips_scores)

# Preprocess frames for FVD
def preprocess_frames_for_fvd(frames):
    frames = np.stack([f.squeeze().cpu().numpy().transpose(0, ) for f in frames], axis=0)
    frames = tf.image.resize(frames, (224, 224))
    frames = tf.cast(frames, tf.float32)
    return frames[None, :]

# def compute_fvd_in_batches(gt_videos, gen_videos, batch_size=16):
#     fvd_scores = []
#     num_batches = len(gt_videos) // batch_size + (len(gt_videos) % batch_size != 0)

#     for i in range(num_batches):
#         start = i * batch_size
#         end = min((i + 1) * batch_size, len(gt_videos))

#         gt_batch = tf.concat(gt_videos[start:end], axis=0)
#         gen_batch = tf.concat(gen_videos[start:end], axis=0)

#         print(tf.reduce_min(gt_batch), tf.reduce_min(gen_batch))
#         print(tf.reduce_max(gt_batch), tf.reduce_max(gen_batch))

#         import pdb; pdb.set_trace()

#         gt_embeddings = i3d_model.signatures["default"](gt_batch)["default"]
#         gen_embeddings = i3d_model.signatures["default"](gen_batch)["default"]

#         mu1, sigma1 = np.mean(gt_embeddings, axis=0), np.cov(gt_embeddings, rowvar=False)
#         mu2, sigma2 = np.mean(gen_embeddings, axis=0), np.cov(gen_embeddings, rowvar=False)

#         diff = mu1 - mu2
#         covmean = sqrtm(sigma1.dot(sigma2))
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real

#         fvd_value = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
#         fvd_scores.append(fvd_value)

#     return np.mean(fvd_scores)  # Average across batches


# Process all images
image_names = sorted(os.listdir(gt_folder))
psnr_results, ssim_results, lpips_results = [], [], []
gt_videos, gen_videos = [], []

for img_name in tqdm(image_names):
    gt_frames = load_and_split_image(os.path.join(gt_folder, img_name))
    gen_frames = load_and_split_image(os.path.join(gen_folder, img_name))
    
    psnr_results.append(compute_psnr(gt_frames, gen_frames))
    ssim_results.append(compute_ssim(gt_frames, gen_frames))
    lpips_results.append(compute_lpips(gt_frames, gen_frames))
    
    # gt_videos.append(preprocess_frames_for_fvd(gt_frames))
    # gen_videos.append(preprocess_frames_for_fvd(gen_frames))

# Print results
print(f"Average PSNR: {np.mean(psnr_results):.4f}")
print(f"Average SSIM: {np.mean(ssim_results):.4f}")
print(f"Average LPIPS: {np.mean(lpips_results):.4f}")

# Compute FVD
# gt_videos = tf.concat(gt_videos, axis=0)
# gen_videos = tf.concat(gen_videos, axis=0)
# fvd_score = compute_fvd_in_batches(gt_videos, gen_videos, batch_size=4)
# print(f"FVD Score: {fvd_score:.4f}")

# Output:
# Average SSIM: 0.1563
# Average LPIPS: 0.5949
# FVD Score: 0.0000