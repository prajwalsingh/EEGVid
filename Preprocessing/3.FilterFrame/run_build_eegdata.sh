# python filterframes_eegvideo_data.py \
# --data_root 'examples/videos/gt' \
# --out_root '../../../dataset/SEED/data/video_frames/' \
# --config_path 'configs/inference/config_4z.yaml' \
# --chunk_size 8 --resolution 256 453

# python filterframes_eegvideo_data.py \
# --data_root 'examples/videos/gt' \
# --out_root '../../../dataset/bollyholly/segment_notch_50Hz_bandpassfilter_0_75Hz_len_2s_overlap_0s_downsampling_200_ssp_False_baseline_None/video/bolly_video_with_z_embedding/' \
# --config_path 'configs/inference/config_4z.yaml' \
# --chunk_size 8 --resolution 720 1280

# python filterframes_eegvideo_data.py \
# --data_root 'examples/videos/gt' \
# --out_root '../../../dataset/bollyholly/segment_notch_50Hz_bandpassfilter_0_75Hz_len_2s_overlap_0s_downsampling_200_ssp_False_baseline_None/video/holly_video_with_z_embedding/' \
# --config_path 'configs/inference/config_4z.yaml' \
# --chunk_size 8 --resolution 720 1280

# python filterframes_eegvideo_data.py \
# --data_root 'examples/videos/gt' \
# --out_root '../../../dataset/DEAP_segmented/len_4s_overlap_0s_ssp_False_baseline_None/video_with_z_embedding/' \
# --config_path 'configs/inference/config_4z.yaml' \
# --chunk_size 8 --resolution 720 1280

# python filterframes_eegvideo_data.py \
# --data_root 'examples/videos/gt' \
# --out_root '../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200_featvec/video_frames/' \
# --config_path 'configs/inference/config_4z.yaml' \
# --chunk_size 8 --resolution 256 453


python filterframes_eegvideo_data.py \
--data_root 'examples/videos/gt' \
--out_root '../../../dataset/SEED-DV/Video/video_frames_final/' \
--config_path 'configs/inference/config_4z.yaml' \
--chunk_size 8 --resolution 256 453