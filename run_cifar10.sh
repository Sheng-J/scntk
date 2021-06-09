python main.py --exp_type accuracy  --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_seeds 2 --n_permutes 100 --n_tests 20 \
    --n_samples 2000 --batch_size 100 \
    --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None \
    --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st2 st2 st2 st2 --writer_record 0
