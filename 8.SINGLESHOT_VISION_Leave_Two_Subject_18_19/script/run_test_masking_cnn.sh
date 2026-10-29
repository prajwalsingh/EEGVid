# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask_type nomask \
#     --network_type cnn \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type channels \
#     --mask_channels noseback_left noseback_right noseback_center\

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type channels \
#     --mask_channels noseback_right noseback_center \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type channels \
#     --mask_channels noseback_left noseback_center \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type channels \
#     --mask_channels noseback_left noseback_right\

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type time \
#     --time_range 0 110 \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type time \
#     --time_range 110 220 \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type time \
#     --time_range 220 330 \


# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type time \
#     --time_range 330 440 \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type time \
#     --time_range 0 220 \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type time \
#     --time_range 220 440 \


### Channel Masking ######

# for n in {0..127}; 
# do
#     python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
#     --network_type cnn \
#     --mask_type leaveone \
#     --keep_channel $n \

# done

### Channel Masking Range ######

for n in {0..127}; 
do
    python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
        --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
        --exp_type all \
        --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
        --split_type normal\
        --eeg_channels 128 \
        --num_classes 40\
        --batch_size 256 \
        --hidden_unit 128 \
        --feat_dims 128 \
        --lstm_depth 4 \
        --epochs 3001 \
        --val_freq 1 \
        --num_workers 4 \
        --mask \
        --network_type cnn \
        --mask_type maskone \
        --chan_range $n \

done


# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth" \
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
# --network_type cnn \
#     --mask_type masksome \
#     --chan_range 6 127 \

# python maskandtest.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/36_EXPERIMENT_all_normal_cnn/bestckpt/eeg_all_feat_0.996.pth"\
#     --split_type normal\
#     --eeg_channels 128 \
#     --num_classes 40\
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --mask \
# --network_type cnn \
#     --mask_type nonbest_channels \