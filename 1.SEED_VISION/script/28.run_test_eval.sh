# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type all \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_left \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_left_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_right \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_right_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_front \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_front_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_back \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_back_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_Q1_right \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_Q1_right_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_Q2_left \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_Q2_left_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_Q3_left \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_Q3_left_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_Q4_right \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_Q4_right_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type noseback_center \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_noseback_center_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type leftrightear_center \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_leftrightear_center_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_frontal_left \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_frontal_left_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_frontal_right \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_frontal_right_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_parietal_left \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_parietal_left_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_parietal_right \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_parietal_right_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_occipital_left \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_occipital_left_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_occipital_right \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_occipital_right_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type lobes_temporal_left \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_temporal_left_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type lobes_temporal_right \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_temporal_right_1/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_frontal \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_frontal_2/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_parietal \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_parietal_2/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
    --exp_type lobes_occipital \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_occipital_2/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 15 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

# python 3.inference_eeg2feat.py --test_path "../../../../dataset/SEED/data/eeg_video_dataset_random_split_v2_tokens/repeat_*/test/" \
#     --exp_type lobes_temporal \
#     --ckpt_path "EXPERIMENT_EEG2FeatAbl_lobes_temporal_2/bestckpt"\
#     --eeg_timesteps 400 \
#     --num_classes 15 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 300 \
#     --val_freq 1 \
#     --num_workers 8