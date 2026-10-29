# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
from typing import List, Optional

import cv2
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import config

import legacy
from training.dataset import EEG2ImageDataset
from videoeeg_dataset import VideoEEGDataset
from tqdm import tqdm
import torchvision

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

#----------------------------------------------------------------------------
def create_collage(ground_truth, predicted, save_path=None, dpi=300):
    fig = plt.Figure(figsize=(10, 10), dpi=dpi)
    ground_truth = ground_truth.permute(1, 2, 3, 0).detach().cpu().numpy()
    predicted = predicted.permute(1, 2, 3, 0).detach().cpu().numpy()
    
    blank_image = np.zeros((8*128, 2*128, 3), np.uint8)

    for loc_idx, (gt_image, pred_image) in enumerate(zip(ground_truth, predicted)):
        gt_image = np.uint8(cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR))
        pred_image = np.uint8(cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR))    
        blank_image[loc_idx*128:loc_idx*128 + 128, :128] = gt_image
        blank_image[loc_idx*128:loc_idx*128 + 128, 128:] = pred_image
    
    cv2.imwrite(save_path, blank_image)
    

def setup_snapshot_image_grid(test_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // test_set.image_shape[2], 8, 32) # For each class 32 samples
    gh = np.clip(5120 // test_set.image_shape[1], 5, 40) # Number of classes 40

    # No labels => show random subset of training samples.
    if not test_set.has_labels:
        all_indices = list(range(len(test_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(test_set)):
            # label = tuple(test_set.get_details(idx).raw_label.flat[::-1])
            label = tuple([test_set.labels[idx].detach().cpu().item()])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)
        
        # print(label_groups)
        # print(len(label_groups.keys()))

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[test_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def save_video(tensor, save_path, fps: float):

    tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
    arr = tensor.detach().cpu().squeeze().to(torch.uint8)
    c, t, h, w = arr.shape
    torchvision.io.write_video(save_path, arr.permute(1, 2, 3, 0), fps=fps, options={'codec': 'libx264', 'crf': '15'})

def encode_position_vectorized(x, enc_dim):
    # Keep original x as first dimension
    positions = [x]
    
    # Create power vector: [0, 1, 2, ..., enc_dim-1]
    powers = torch.arange(enc_dim, device=x.device)
    
    # Calculate frequencies: [2^0, 2^1, 2^2, ..., 2^(enc_dim-1)]
    frequencies = torch.pow(2.0, powers)
    
    # Expand dimensions for broadcasting
    # x shape becomes compatible with frequencies
    x_expanded = x.unsqueeze(-1)
    
    # Multiply x by all frequencies at once
    xf = x_expanded * frequencies
    
    # Calculate sin and cos for all frequencies simultaneously
    sin_encodings = torch.sin(xf)
    cos_encodings = torch.cos(xf)
    
    # Interleave sin and cos values
    encodings = torch.stack([sin_encodings, cos_encodings], dim=-1).flatten(-2)
    
    # Concatenate original x with encodings along the last dimension
    result = torch.cat([x.unsqueeze(-1), encodings], dim=-1)
    
    return result.to('cuda')


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    data: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # os.makedirs(outdir+'/{}'.format(class_idx), exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    x_dataset = VideoEEGDataset(data,split_type='val', load_frames=True)
    # test_set_iterator = torch.utils.data.DataLoader(dataset=test_set, sampler=test_set_sampler, batch_size=args.batch_size)
    batch_gpu = 4
    total_frames = 8
    
    resize_transform = torchvision.transforms.Resize((config.image_height, config.image_width))

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        # grid_size, images, labels = setup_snapshot_image_grid(test_set=test_set, random_seed=seed)
        # grid_size, images, labels = setup_snapshot_image_grid(test_set=test_set) # Uncomment if using random noise generation
        for video_idx, (eeg_feat, class_id, video_frames_path, file_name) in enumerate(tqdm(x_dataset)):

            eeg_feat = eeg_feat.to(device)
            # video_frames_gt = torchvision.transforms.Resize(size=(128, 128))(video_frames_gt)
            video_data = torch.tensor(np.load(video_frames_path, allow_pickle=True)['frames']).float()
            video_frames_gt = resize_transform(video_data)

            video_frames_gen = []
            grid_z = torch.randn([1, G.z_dim], device=device)#.split(batch_gpu)

            for frame_idx in range(8):
                # temp_eeg_feat = torch.nn.functional.one_hot(torch.tensor(frame_idx), num_classes=total_frames).float().to(device)
                temp_eeg_feat = encode_position_vectorized(torch.tensor(frame_idx).float(), 10)
                grid_c = torch.cat([eeg_feat, temp_eeg_feat], dim=0).unsqueeze(0)
                images = G(z=grid_z, c=grid_c, noise_mode='const').cpu().squeeze(0)
                video_frames_gen.append(images)

            video_frames_gen = torch.stack(video_frames_gen, dim=0).permute(1, 0, 2, 3)

            # save_video(video_frames_gt, f'generated_videos/{video_idx}_gt.mp4', fps=4)
            # save_video(video_frames_gen, f'generated_videos/{video_idx}_pred.mp4', fps=4)
            video_frames_gt = torch.clamp((video_frames_gt + 1) / 2, 0, 1) * 255
            video_frames_gen = torch.clamp((video_frames_gen + 1) / 2, 0, 1) * 255

            create_collage(video_frames_gt, video_frames_gen, save_path=f'generated_videos/{file_name}_vidx_{video_idx}.png', dpi=300)

        # images = torch.cat([G(z=z, c=c, noise_mode='random').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        # save_image_grid(images, f'{outdir}/seed{seed:04d}.pdf', drange=[-1,1], grid_size=grid_size)

    # Synthesize the result of a W projection.
    # if projected_w is not None:
    #     if seeds is not None:
    #         print ('warn: --seeds is ignored when using --projected-w')
    #     print(f'Generating images from projected W "{projected_w}"')
    #     ws = np.load(projected_w)['w']
    #     ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
    #     assert ws.shape[1:] == (G.num_ws, G.w_dim)
    #     for idx, w in enumerate(ws):
    #         img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
    #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #         img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
    #     return

    # if seeds is None:
    #     ctx.fail('--seeds option is required when not using --projected-w')

    # # Labels.
    # label = torch.zeros([1, G.c_dim], device=device)
    # if G.c_dim != 0:
    #     if class_idx is None:
    #         ctx.fail('Must specify class label with --class when using a conditional network')
    #     label[:, class_idx] = 1
    # else:
    #     if class_idx is not None:
    #         print ('warn: --class=lbl ignored when running on an unconditional network')

    # # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    #     z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    #     img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #     PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{class_idx}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
