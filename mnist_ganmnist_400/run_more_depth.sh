#!/bin/bash
#SBATCH --partition=p100
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd $HOME/scntk_icml
list=(
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st3 st3 st3 st3"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st3 st3 st2 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 3 3 --poolings st2 st2 st2 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 2 2 --poolings st3 st3 st3 st3"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 2 2 --poolings st3 st3 st2 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 300 --g_activs  cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1 1 1 --filter_sizes 3 3 2 2 --poolings st2 st2 st2 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st3 st3 st3"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st3 st3 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 3 --poolings st2 st2 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 2 --poolings st3 st3 st3"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 2 --poolings st3 st3 st2"
  "python main.py --exp_type accuracy --S_P_data MNIST --S_Q_data GANMNIST --n_samples 400 --batch_size 100 --widths 300 300 300 --g_activs  cos relu relu --bandwidths 1.0 None None --init_stdvs None 1 1 --filter_sizes 3 3 2 --poolings st2 st2 st2"
)
${list[SLURM_ARRAY_TASK_ID]} 
