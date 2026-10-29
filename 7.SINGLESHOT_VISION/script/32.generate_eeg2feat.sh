python 7.generate_eeg2feat.py --test_path "" \
    --exp_type all \
    --ckpt_path "EXPERIMENT_EEG2FeatAbl_all_1/bestckpt"\
    --eeg_timesteps 400 \
    --num_classes 13 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8