# Flatness-Aware Stochastic Gradient Langevin Dynamics (fSGLD)

**This codebase is built upon the [Entropy-MCMC]((https://openreview.net/forum?id=oGNdBvymod)) implementation.** We replicated their code and added our novel **fSGLD** method on top of it.

## Requirements

### Recommended Environment
```bash
python==3.8
pytorch==1.12
torchvision
numpy
tqdm
```

### Installation
```bash
pip install torch==1.12.0 torchvision numpy tqdm
```


## Quick Start

### Running fSGLD on CIFAR-10

The simplest way to reproduce our results is using the provided script:

```bash
bash run_fsgld.sh
```

### Manual Execution

You can also run experiments manually with custom hyperparameters:

```bash
# CIFAR-10
python exp/cifar_fsgld.py \
    --num-class 10 \
    --save cifar10_fsgld_custom \
    --sigma 0.001 \
    --temperature 1e-4 \
    --gpu 0 \
    --seed 42

# CIFAR-100
python exp/cifar_fsgld.py \
    --num-class 100 \
    --save cifar100_fsgld_custom \
    --sigma 0.001 \
    --temperature 1e-4 \
    --gpu 0 \
    --seed 42
```

## Experimental Results

When you run experiments, results are automatically saved to `.checkpoints/{save_name}_seed{seed}_{date}/test_results_seed{seed}.txt`.

Each result file contains:

**Experiment Configuration:**
- Dataset, save path, random seed
- Training hyperparameters: epochs, batch size, learning rate, sigma, temperature, decay scheme

**Performance Metrics:**
- `ACC (%)`: Test accuracy
- `NLL`: Negative Log-Likelihood
- `AUROC (%)`: Area Under ROC Curve for in-distribution uncertainty
- `ECE (%)`: Expected Calibration Error
- `OOD AUROC (%)`: Out-of-distribution detection AUROC
- `OOD AUPR (%)`: Out-of-distribution detection AUPR

Example output format:
```
==================================================
EXPERIMENT CONFIGURATION
==================================================
Dataset: CIFAR-10
Seed: 178
Sigma: 0.001
Temperature: 0.0001
...
==================================================

ACC   (%): 95.81
NLL      : 0.143
AUROC (%): 93.50
ECE   (%): 0.68
OOD AUROC (%): 97.95
OOD AUPR  (%): 98.76
```

