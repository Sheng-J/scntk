#!/bin/bash
#SBATCH --partition=p100
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19
cd $HOME/scntk_icml
list=(
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 200 200 200 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 200 200 200 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st2 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st2 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st1 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st1 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st4"
  "python main.py --n_seeds 2 --n_tests 20 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st4 st4 st4"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 200 200 200 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 200 200 200 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st2 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st2 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st1 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st1 st2 st2 st2"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st4"
  "python main.py --n_seeds 2 --n_tests 60 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 1000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st4 st4 st4"
)
${list[SLURM_ARRAY_TASK_ID]} 
