python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt"\
    --eeg_timesteps 400 \
    --mask_t_start 0 \
    --mask_t_end 100 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt"\
    --eeg_timesteps 400 \
    --mask_t_start 100 \
    --mask_t_end 200 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt"\
    --eeg_timesteps 400 \
    --mask_t_start 200 \
    --mask_t_end 300 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt"\
    --eeg_timesteps 400 \
    --mask_t_start 300 \
    --mask_t_end 400 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8