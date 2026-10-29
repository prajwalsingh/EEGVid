python maskandfilter.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
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
    --mask \
    --mask_type custom \
    --task plot \

# python maskandfilterdp.py --train_path "../../data/eeg_imagenet40_cvpr_2017_raw/train/*" \
#     --test_path "../../data/eeg_imagenet40_cvpr_2017_raw/val/*" \
#     --exp_type all \
#     --ckpt_path "./Run/1_EXPERIMENT_/bestckpt/eeg_all_feat_0.9642857142857143.pth"\
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
#     --mask_type custom \