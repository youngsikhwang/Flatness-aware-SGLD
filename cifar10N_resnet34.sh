#!/bin/bash
#SBATCH -J cifar10N_resnet34
#SBATCH -p a100_40g
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time=12:00:00

module purge
export CONDA_HOME=/apps/applications/miniconda3
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate fgld

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Starting CIFAR10N + ResNet34 with 4 optimizers"
echo "Time: $(date)"

CUDA_VISIBLE_DEVICES=0 python main_auto.py \
    --dataset cifar10N_cifar \
    --backbone resnet34 \
    --optimizer sgd \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar10N_resnet34_sgd \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=1 python main_auto.py \
    --dataset cifar10N_cifar \
    --backbone resnet34 \
    --optimizer sam \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar10N_resnet34_sam \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=2 python main_auto.py \
    --dataset cifar10N_cifar \
    --backbone resnet34 \
    --optimizer sgld \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar10N_resnet34_sgld \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=3 python main_auto.py \
    --dataset cifar10N_cifar \
    --backbone resnet34 \
    --optimizer fgld \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar10N_resnet34_fgld \
    --device cuda \
    --seed 42 \
    --visualize &


CUDA_VISIBLE_DEVICES=4 python main_auto.py \
    --dataset cifar100N_cifar \
    --backbone resnet34 \
    --optimizer sgd \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet34_sgd \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=5 python main_auto.py \
    --dataset cifar100N_cifar \
    --backbone resnet34 \
    --optimizer sam \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet34_sam \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=6 python main_auto.py \
    --dataset cifar100N_cifar \
    --backbone resnet34 \
    --optimizer sgld \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet34_sgld \
    --device cuda \
    --seed 42 \
    --visualize &

CUDA_VISIBLE_DEVICES=7 python main_auto.py \
    --dataset cifar100N_cifar \
    --backbone resnet34 \
    --optimizer fgld \
    --n_trials 20 \
    --epochs 150 \
    --save_dir ./optuna_results/cifar100N_resnet34_fgld \
    --device cuda \
    --seed 42 \
    --visualize &



echo "All optimizers started. Waiting for completion..."
wait
