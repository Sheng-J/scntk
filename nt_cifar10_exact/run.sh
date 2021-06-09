#!/bin/bash
#SBATCH --partition=p100,t4v2,rtx6080
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd $HOME/scntk_icml
list=(
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st2 st2 st2"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 4 4 4 --poolings st2 st2 st2"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st3 st3 st3"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st3 st3 st3 st3"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st2 st2 st2 st2"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st3 st3 st3 st3"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 400 100 100 100 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st3 st3 st3 st3"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 400 100 100 100 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st2 st2 st2 st2"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st1 st3 st3 st3"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st1 st2 st2 st2"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 4 4 4 4 --poolings st1 st3 st3 st3"
  "python main.py --kernel_type exact_ntk --n_seeds 10 --n_tests 100 --exp_type accuracy --S_P_data CIFAR10 --S_Q_data CIFAR10_1 --n_samples 2000 --batch_size 100 --widths 400 100 100 100 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st1 st3 st3 st3"
)
${list[SLURM_ARRAY_TASK_ID]} 
