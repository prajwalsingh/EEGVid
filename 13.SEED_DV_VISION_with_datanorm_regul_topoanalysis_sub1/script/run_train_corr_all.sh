python check_correlation.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --exp_type all \
    --split_type normal\
    --eeg_channels 128 \
    --num_classes 40\
    --batch_size 1 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 3001 \
    --val_freq 1 \
    --num_workers 4 \
    