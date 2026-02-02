#!/bin/bash

echo "=== Running fSGLD Experiments ==="

# Define seeds for 3 runs
seeds=(54 63 178)

for seed in "${seeds[@]}"; do
    echo "CIFAR-10 Seed: $seed"
    python exp/cifar_fsgld.py --num-class 10 --save cifar10_fsgld_s0.001_t1e-4 --sigma 0.001 --temperature 1e-4 --gpu 0 --seed $seed 
done


echo "=== All experiments completed ==="
