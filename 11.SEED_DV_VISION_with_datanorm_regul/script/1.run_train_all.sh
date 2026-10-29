python 1.eeg2feat.py --eeg_path "../../../../dataset/SEED-DV/EEG/*.npy" \
    --exp_type all \
    --eeg_timesteps 400 \
    --num_classes 40 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8