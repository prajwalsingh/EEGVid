# python 1.eeg2feat.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
#     --exp_type all \
#     --subject "sub1" \
#     --data_aug 'timerev' \
#     --eeg_timesteps 400 \
#     --num_classes 40 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 200 \
#     --val_freq 1 \
#     --num_workers 8

# python 1.eeg2feat.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
#     --exp_type all \
#     --subject "sub1" \
#     --data_aug 'gauss' 'timemask'\
#     --eeg_timesteps 400 \
#     --num_classes 40 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 200 \
#     --val_freq 1 \
#     --num_workers 8

# python 1.eeg2feat.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
#     --exp_type all \
#     --subject "sub1" \
#     --data_aug 'gauss' \
#     --eeg_timesteps 400 \
#     --num_classes 40 \
#     --batch_size 256 \
#     --feat_dims 1024 \
#     --epochs 200 \
#     --val_freq 1 \
#     --num_workers 8


python 1.eeg2feat.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
    --exp_type all \
    --subject "sub1" \
    --data_aug 'timemask'\
    --eeg_timesteps 400 \
    --num_classes 40 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 200 \
    --val_freq 1 \
    --num_workers 8