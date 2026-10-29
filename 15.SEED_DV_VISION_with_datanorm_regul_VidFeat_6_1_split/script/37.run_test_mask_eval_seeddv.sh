
python 11.eeg2feat_timemask_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
    --exp_type all \
    --data_aug 'timerev' 'timemask'\
    --eeg_timesteps 400 \
    --mask_t_start 0 \
    --mask_t_end 100 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 11.eeg2feat_timemask_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
    --exp_type all \
    --data_aug 'timerev' 'timemask'\
    --eeg_timesteps 400 \
    --mask_t_start 100 \
    --mask_t_end 200 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 11.eeg2feat_timemask_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
    --exp_type all \
    --data_aug 'timerev' 'timemask'\
    --eeg_timesteps 400 \
    --mask_t_start 200 \
    --mask_t_end 300 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8

python 11.eeg2feat_timemask_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
    --exp_type all \
    --data_aug 'timerev' 'timemask'\
    --eeg_timesteps 400 \
    --mask_t_start 300 \
    --mask_t_end 400 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8
