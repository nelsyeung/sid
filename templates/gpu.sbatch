#!/bin/bash
#SBATCH --job-name=sid
#SBATCH --output=sid.out

#SBATCH --partition=pnode
#SBATCH --gres=gpu:P100:2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64371

export PNODE=/home/theory/user/pnode
export PATH=$PNODE/bin:$PATH
export LD_LIBRARY_PATH=$PNODE/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PNODE/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PNODE/lib/python2.7
export CPATH=$PNODE/include:$CPATH
export PKG_CONFIG_PATH=$PNODE/lib/pkgconfig:$PKG_CONFIG_PATH
export CUDA_HOME=$PNODE/cuda-9.1
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE

GPUS=2 python2 train.py
