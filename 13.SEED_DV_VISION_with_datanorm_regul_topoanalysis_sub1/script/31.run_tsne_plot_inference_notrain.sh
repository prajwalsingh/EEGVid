python 6.tsne_eeg2feat_notrain.py --test_path "../../../../dataset/Preprocessed_SingleShot/segment_notch_50Hz_bandpassfilter_0_5_NoneHz_len_2s_overlap_0s_downsampling_200/eeg/test/" \
    --exp_type all \
    --save_path "EXPERIMENT_EEG2FeatAbl_all_1/notrain_tsne_plot/" \
    --vid_labels 13 \
    --eeg_timesteps 400 \
    --batch_size 256 \
    --feat_dims 1024 \
    --epochs 300 \
    --val_freq 1 \
    --num_workers 8