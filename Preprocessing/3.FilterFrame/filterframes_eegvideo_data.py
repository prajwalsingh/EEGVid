import os
import numpy as np
from natsort import natsorted
from glob import glob
from tqdm import tqdm

import torch
import cv2
import argparse
import logging
from decord import VideoReader, cpu
from glob import glob
from PIL import Image
from omegaconf import OmegaConf
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Video VAE Inference Script")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the folder containing input videos.",
    )
    parser.add_argument(
        "--out_root", type=str, required=True, help="Path to save reconstructed videos."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Number of frames per chunk for processing.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Resolution to process videos (height, width).",
    )
    return parser.parse_args()


def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def data_processing(video_path, resolution):
    """Load and preprocess video data."""
    try:
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_resolution = video_reader[0].shape

        # Rescale resolution to match specified limits
        resolution = [
            min(video_resolution[0], resolution[0]),
            min(video_resolution[1], resolution[1]),
        ]
        video_reader = VideoReader(
            video_path, ctx=cpu(0), width=resolution[1], height=resolution[0]
        )

        video_length = len(video_reader)
        vid_fps = video_reader.get_avg_fps()
        frame_indices = list(range(0, video_length))
        frames = video_reader.get_batch(frame_indices)
        assert (
            frames.shape[0] == video_length
        ), f"Frame mismatch: {len(frames)} != {video_length}"

        frames = (
            torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        )  # [t, h, w, c] -> [c, t, h, w]
        frames = (frames / 255 - 0.5) * 2  # Normalize to [-1, 1]
        return frames, vid_fps
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return None, None


def save_video(tensor, save_path, fps: float):
    """Save video tensor to a file."""
    try:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        arr = tensor.detach().cpu().squeeze().to(torch.uint8)
        c, t, h, w = arr.shape
            
        torchvision.io.write_video(save_path, arr.permute(1, 2, 3, 0), fps=fps, options={'codec': 'libx264', 'crf': '15'})
        logging.info(f"Video saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving video {save_path}: {e}")


def filter_frames_by_optical_flow(video_tensor, num_frames=8, save_path=None, fps=30):
    """
    Filter frames from a video tensor based on maximum optical flow change and optionally save as a video.
    
    Args:
        video_tensor (torch.Tensor): Input tensor of shape [C, T, H, W], e.g., [3, 48, 720, 1280]
        num_frames (int): Number of frames to select (default: 8)
        save_path (str, optional): Path to save the filtered frames as a video (e.g., 'output.mp4')
        fps (int): Frames per second for the output video (default: 30)
    
    Returns:
        torch.Tensor: Tensor of shape [C, num_frames, H, W] with selected frames
    """
    # Validate input
    if video_tensor.ndim != 4 or video_tensor.shape[1] < num_frames:
        raise ValueError("Invalid input tensor shape or insufficient frames.")
    
    # Convert tensor to NumPy: [T, H, W, C] for OpenCV, permute from [C, T, H, W]
    video_np = np.clip(((video_tensor.permute(1, 2, 3, 0).numpy() + 0.5) * 0.5)*255.0, 0, 255)  # Shape: [48, 720, 1280, 3]
    
    # Initialize list to store optical flow magnitudes
    flow_magnitudes = []

    # Convert first frame to grayscale for optical flow
    prev_frame = cv2.cvtColor(video_np[0].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Compute optical flow for each pair of consecutive frames
    for t in range(1, video_np.shape[0]):
        curr_frame = cv2.cvtColor(video_np[t].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute dense optical flow using Farneback
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, 
                                            flags=0)
        
        # Calculate magnitude of flow vectors
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_magnitude = np.mean(flow_magnitude)
        flow_magnitudes.append(mean_magnitude)
        
        # Update previous frame
        prev_frame = curr_frame
    
    # Append a zero for the first frame (no flow) to match length
    flow_magnitudes = [0] + flow_magnitudes
    
    # Get indices of top num_frames frames based on flow magnitude
    selected_indices = np.argsort(flow_magnitudes)[-num_frames:]
    selected_indices = np.sort(selected_indices)  # Sort for temporal order
    
    # Extract selected frames from original tensor
    filtered_tensor = video_tensor[:, selected_indices, :, :]
    
    # Save filtered frames as a video if save_path is provided
    if save_path:
        # Initialize video writer
        height, width = video_tensor.shape[2], video_tensor.shape[3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Convert filtered frames to NumPy and write to video
        filtered_np = np.clip(((filtered_tensor.permute(1, 2, 3, 0).numpy() + 0.5) * 0.5)*255.0, 0, 255)  # [num_frames, H, W, C]
        for frame in filtered_np:
            frame_uint8 = frame.astype(np.uint8)  # Ensure uint8 for OpenCV
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            out.write(frame_bgr)
        
        # Release video writer
        out.release()
        print(f"Filtered frames saved as video at: {save_path}")
    
    return filtered_tensor


if __name__ == '__main__':
    
    args = parse_args()

    # video_paths = natsorted(glob('../../../dataset/SEED/data/video_chunks_reshape/*'))
    # video_paths = natsorted(glob('../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/video/*'))
    video_paths = natsorted(glob('../../../dataset/SEED-DV/Video/extracted_clips/*'))

    config = OmegaConf.load(args.config_path)

    os.makedirs(args.out_root, exist_ok=True)

    image_transform = clip_transform(224)

    # Process each video
    for video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
        logging.info(f"Processing video: {video_path}")
        gt_frames, vid_fps = data_processing(video_path, args.resolution)
        if gt_frames is None:
            continue
        
        gt_frames = filter_frames_by_optical_flow(gt_frames, num_frames=8)

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        gt_frames = gt_frames.squeeze(dim=0).cpu().numpy()

        save_path = os.path.join(args.out_root, f"{video_name}.npz")

        np.savez_compressed(save_path, frames=gt_frames)
