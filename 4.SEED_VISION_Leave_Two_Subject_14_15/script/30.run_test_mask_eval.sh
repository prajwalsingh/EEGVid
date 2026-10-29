python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens_leave_two_subject/repeat_*/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt/eegfeat_all_0.21764705882352942.pth"\
    --eeg_timesteps 400 \
    --mask_t_start 0 \
    --mask_t_end 100 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens_leave_two_subject/repeat_*/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt/eegfeat_all_0.21764705882352942.pth"\
    --eeg_timesteps 400 \
    --mask_t_start 100 \
    --mask_t_end 200 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens_leave_two_subject/repeat_*/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt/eegfeat_all_0.21764705882352942.pth"\
    --eeg_timesteps 400 \
    --mask_t_start 200 \
    --mask_t_end 300 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 4.inference_mask_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens_leave_two_subject/repeat_*/test/" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt/eegfeat_all_0.21764705882352942.pth"\
    --eeg_timesteps 400 \
    --mask_t_start 300 \
    --mask_t_end 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8