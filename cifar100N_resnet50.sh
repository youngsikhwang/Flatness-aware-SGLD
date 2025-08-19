#!/bin/bash
#SBATCH -J cifar100N_resnet50
#SBATCH -p a100_40g
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --gres=gpu:4
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time=12:00:00

module purge
export CONDA_HOME=/apps/applications/miniconda3
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate fgld


export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Starting CIFAR100N + ResNet50 with 4 optimizers on 4 GPUs..."
echo "Time: $(date)"

CUDA_VISIBLE_DEVICES=0 python main_auto.py \
    --dataset cifar100N \
    --backbone resnet50 \
    --optimizer sgd \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet50_sgd \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=1 python main_auto.py \
    --dataset cifar100N \
    --backbone resnet50 \
    --optimizer sam \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet50_sam \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=2 python main_auto.py \
    --dataset cifar100N \
    --backbone resnet50 \
    --optimizer sgld \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet50_sgld \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=3 python main_auto.py \
    --dataset cifar100N \
    --backbone resnet50 \
    --optimizer fgld \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet50_fgld \
    --device cuda \
    --seed 42 \
    --visualize &

echo "All optimizers started. Waiting for completion..."
wait