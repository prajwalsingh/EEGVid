# subjects=("sub1" "sub2" "sub3" "sub4" "sub5" "sub6" "sub7" "sub8" "sub9" "sub10" "sub11" "sub12" "sub13" "sub14" "sub15" "sub16" "sub17" "sub18" "sub19" "sub20")  # Add more subjects as needed
subjects=("sub1")  # Add more subjects as needed

for subject in "${subjects[@]}"; do
    echo "Running for subject: $subject"

    python 1.eeg2feat.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --exp_type all \
        --subject "$subject" \
        --exp_type noseback_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8
done