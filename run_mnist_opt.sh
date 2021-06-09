python main.py --exp_type accuracy  --S_P_data MNIST --S_Q_data GANMNIST --n_seeds 10 --n_permutes 100 \
    --n_samples 400 --batch_size 100 \
    --widths 400 400 400 400 --g_activs  cos relu relu relu --bandwidths 1.00 None None None \
    --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st3 st3 st3 st3 --writer_record 0
