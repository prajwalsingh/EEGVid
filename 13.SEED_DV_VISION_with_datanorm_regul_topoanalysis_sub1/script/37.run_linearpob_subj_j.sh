subjects=("sub1")  # Add more subjects as needed

for subject in "${subjects[@]}"; do
        echo "Running for subject: $subject"
    
        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
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

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_front \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_back \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_Q2_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_Q3_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_Q1_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_Q4_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type noseback_center \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type leftrightear_center \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_frontal_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_frontal_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_parietal_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_parietal_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_occipital_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_occipital_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_temporal_left \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_temporal_right \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_frontal \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_parietal \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_occipital \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

        python 8.eeg2feat_svmlinear_test.py --eeg_path "../../../../dataset/SEED-DV/EEG/" \
        --subject "$subject" \
        --exp_type lobes_temporal \
        --data_aug 'timerev' 'timemask'\
        --eeg_timesteps 400 \
        --num_classes 40 \
        --batch_size 256 \
        --feat_dims 1024 \
        --epochs 300 \
        --val_freq 1 \
        --num_workers 8

done