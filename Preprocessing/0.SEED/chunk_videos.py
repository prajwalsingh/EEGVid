import os
import pandas as pd
from moviepy.editor import VideoFileClip

# Create a folder to save the video chunks
chunks_folder = "video_chunks"
if not os.path.exists(chunks_folder):
    os.makedirs(chunks_folder)

# Function to chunk the video into 48 frames (2 seconds)
def chunk_video(video_path, experiment_id, emotion_label):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        chunk_duration = 2  # each chunk will be 2 seconds (48 frames at 24fps)
        
        # Calculate number of chunks
        num_chunks = int(duration // chunk_duration)
        
        for chunk_id in range(num_chunks):
            start_time = chunk_id * chunk_duration
            end_time = start_time + chunk_duration
            
            # Create a subclip
            subclip = clip.subclip(start_time, end_time)
            chunk_filename = f"{experiment_id}_{emotion_label}_{chunk_id}.mp4"
            subclip.write_videofile(os.path.join(chunks_folder, chunk_filename), codec="libx264")
            print(f"Created chunk: {chunk_filename}")

    except Exception as e:
        print(f"Failed to chunk video {experiment_id}: {e}")

# Process each trimmed video file
trimmed_videos_folder = "clips"
trimmed_videos = [f for f in os.listdir(trimmed_videos_folder) if f.endswith('.mp4')]

for video_file in trimmed_videos:
    # Extract Experiment ID from the filename
    experiment_id = video_file.split('_')[0]
    emotion_label = video_file.split('_')[1]
    video_path = os.path.join(trimmed_videos_folder, video_file)
    chunk_video(video_path, experiment_id, emotion_label)