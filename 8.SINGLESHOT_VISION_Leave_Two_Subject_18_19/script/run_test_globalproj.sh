python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type all \
    --ckpt_path "./Run/1_EXPERIMENT_/bestckpt/eeg_all_feat_0.9642857142857143.pth"\
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

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_left \
    --ckpt_path "./Run/2_EXPERIMENT_/bestckpt/eeg_left_feat_0.9079241071428571.pth"\
    --split_type normal\
    --eeg_channels 60 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_right \
    --ckpt_path "./Run/3_EXPERIMENT_/bestckpt/eeg_right_feat_0.8984375.pth"\
    --split_type normal\
    --eeg_channels 60 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_front \
    --ckpt_path "./Run/14_EXPERIMENT_noseback_front_normal/bestckpt/eeg_nose_quad_feat_0.929.pth"\
    --split_type normal\
    --eeg_channels 54 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_back \
    --ckpt_path "./Run/15_EXPERIMENT_noseback_back_normal/bestckpt/eeg_back_quad_feat_0.928.pth"\
    --split_type normal\
    --eeg_channels 65 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_Q2_left \
    --ckpt_path "./Run/18_EXPERIMENT_nose_quad_2_left_normal/bestckpt/eeg_nose_quad_2_left_feat_0.855.pth"\
    --split_type normal\
    --eeg_channels 26 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_Q3_left \
    --ckpt_path "./Run/19_EXPERIMENT_back_quad_3_left_normal/bestckpt/eeg_back_quad_3_left_feat_0.89.pth"\
    --split_type normal\
    --eeg_channels 30 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_Q1_right \
    --ckpt_path "./Run/20_EXPERIMENT_nose_quad_1_right_normal/bestckpt/eeg_nose_quad_1_right_feat_0.739.pth"\
    --split_type normal\
    --eeg_channels 26 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_Q4_right \
    --ckpt_path "./Run/21_EXPERIMENT_back_quad_4_right_normal/bestckpt/eeg_back_quad_4_right_feat_0.905.pth"\
    --split_type normal\
    --eeg_channels 30 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type noseback_center \
    --ckpt_path "./Run/22_EXPERIMENT_noseback_center_normal/bestckpt/eeg_noseback_center_feat_0.68.pth"\
    --split_type normal\
    --eeg_channels 8 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \

python global_projection.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --exp_type leftrightear_center \
    --ckpt_path "./Run/23_EXPERIMENT_leftrightear_center_normal/bestckpt/eeg_leftrightear_center_feat_0.892.pth"\
    --split_type normal\
    --eeg_channels 9 \
    --num_classes 40\
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \
