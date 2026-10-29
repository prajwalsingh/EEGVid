python 10.eeg2feat_featsim.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --exp_type all \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8