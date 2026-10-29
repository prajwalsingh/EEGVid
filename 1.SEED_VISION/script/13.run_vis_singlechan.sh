python train_single_chan.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
    --val_path "../../data/eeg_imagenet40_cvpr_2017_raw/test/*" \
    --split_type normal \
    --exp_type one_channel \
    --eeg_channels 1 \
    --num_classes 40 \
    --batch_size 256 \
    --hidden_unit 128 \
    --feat_dims 128 \
    --lstm_depth 4 \
    --epochs 1501 \
    --val_freq 1 \
    --num_workers 4 \
    --network_type lstm \
    --chan_range 1 128 \
    --vis

# python train.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --val_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --split_type random \
#     --eeg_channels 128 \
#     --num_classes 40 \
#     --batch_size 256 \
#     --hidden_unit 128 \
#     --feat_dims 128 \
#     --lstm_depth 4 \
#     --epochs 3001 \
#     --val_freq 1 \
#     --num_workers 4 \
#     --network_type cnn \