#!/bin/bash
#SBATCH --partition=p100,t4v1,t4v2,rtx6000
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd $HOME/NTKTST
list=(
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 3 3 3 3 --poolings st4 st4 st4 st4"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 3 3 3 3 --poolings st2 st2 st2 st2"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 3 3 3 3 --poolings st1 st3 st3 st3"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 4 4 4 4  --poolings st4 st4 st4 st4"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 4 4 4 4  --poolings st2 st2 st2 st2"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 100 --g_activs cos relu relu relu --bandwidths 1.0 None None None --init_stdvs None 1.0 1.0 1.0 --filter_sizes 4 4 4 4  --poolings st1 st3 st3 st3"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 4 4 4  --poolings st4 st4 st4"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 4 4 4  --poolings st2 st2 st2"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 4 4 4  --poolings st1 st3 st3"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 3 3 3  --poolings st4 st4 st4"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 3 3 3  --poolings st2 st2 st2"
    "python NTKMMD_CIFAR10.py --n 1000 --batch_size 100 --widths 100 100 100 --g_activs cos relu relu --bandwidths 1.0 None None --init_stdvs None 1.0 1.0 --filter_sizes 3 3 3  --poolings st1 st3 st3"
)
${list[SLURM_ARRAY_TASK_ID]} 
