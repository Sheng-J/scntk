#!/bin/bash
#SBATCH --partition=p100
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
#SBATCH --account=deadline
#SBATCH --qos=deadline
cd $HOME/NTKTST
list=(
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 3 3 3 3 --poolings st4 st4 st4 st4"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 200 200 200 200 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 3 3 3 3 --poolings st2 st2 st2 st2"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 250 250 250 250 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 3 3 3 3 --poolings st1 st2 st2 st2"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 4 4 4 4 --poolings st4 st4 st4 st4"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 200 200 200 200 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 4 4 4 4 --poolings st2 st2 st2 st2"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 250 250 250 250 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 4 4 4 4 --poolings st1 st2 st2 st2"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 3 3 3 --poolings st4 st4 st4"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 200 200 200 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 3 3 3 --poolings st2 st2 st2"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 250 250 250 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 3 3 3 --poolings st1 st2 st4"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 4 4 4 --poolings st4 st4 st4"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 200 200 200 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 4 4 4 --poolings st2 st2 st2"
    "python NTKMMD_CIFAR10.py --K 20 --N 20 --n 1000 --batch_size 100 --widths 250 250 250 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 4 4 4 --poolings st1 st2 st4"
)
${list[SLURM_ARRAY_TASK_ID]} 
