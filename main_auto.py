import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet34, resnet50
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from typing import Dict, Any, Optional
import json
import logging
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly

from models import *
from optimizers import *
from dataloader import get_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_backbone(backbone_name: str, num_classes: int):
    """Get backbone architecture"""
    
    if backbone_name == 'resnet34':
        model = resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'resnet34_cifar':
        model = ResNet34(num_classes)            
    elif backbone_name == 'resnet50_cifar':
        model = ResNet50(num_classes)    
    elif backbone_name == 'vit_b_16':
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.heads.parameters():
            p.requires_grad = True
        for blk in model.encoder.layers:
            for p in blk.self_attention.parameters():
                p.requires_grad = True    
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    return model


def get_optimizer(model, optimizer_name: str, lr: float, **kwargs):
    """Get optimizer"""
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=kwargs.get('milestones', [50, 100]), 
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),  # (beta1, beta2)
            weight_decay=kwargs.get('weight_decay', 1e-2),
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=kwargs.get('milestones', [50, 100]), 
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif optimizer_name == 'fsgld':
        optimizer = fSGLD(
            model.parameters(),
            lr=lr,
            sigma=kwargs.get('sigma', 0.001),
            n_pert=kwargs.get('n_pert', 1),
            momentum=0.0,
            weight_decay=kwargs.get('weight_decay', 5e-4),
            beta_inv=kwargs.get('beta_inv', 1e-14),
            pert_type=kwargs.get('pert_type', 'normal'),
            antithetic=kwargs.get('antithetic', False),
            beta_coupling=kwargs.get('beta_coupling', False),
            eta=kwargs.get('eta', 0.01)
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=kwargs.get('milestones', [50, 100]), 
            gamma=kwargs.get('gamma', 0.1)
        )

    elif optimizer_name == 'sgld':
        optimizer = SGLD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.0),
            weight_decay=kwargs.get('weight_decay', 5e-4),
            beta_inv=kwargs.get('beta_inv', 1e-14)
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=kwargs.get('milestones', [50, 100]), 
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif optimizer_name == 'sam':
        base_opt = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 5e-4)
        )
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_opt,
            rho=kwargs.get('rho', 0.05),
            adaptive=kwargs.get('adaptive', False)
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            base_opt, 
            milestones=kwargs.get('milestones', [50, 100]), 
            gamma=kwargs.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, optimizer_name):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if len(batch_data) == 2:
            data, target = batch_data
        elif len(batch_data) == 3:
            data, target, _ = batch_data
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        
        data, target = data.to(device), target.to(device)
        
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss, output
        
        if optimizer_name in ['fsgld','sgld','sam']:
            loss, output = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 2:
                data, target = batch_data
            elif len(batch_data) == 3:
                data, target, _ = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def objective(trial: optuna.Trial, args):
    """Optuna objective function for hyperparameter optimization"""
    
    # Fixed parameters
    batch_size = 128  # Fixed batch size
    weight_decay = 5e-4  # Fixed weight decay
    
    # Optimizer-specific hyperparameter tuning
    optimizer_kwargs = {
        'weight_decay': weight_decay,
        'milestones': [50, 100],
        'gamma': 0.1  # Fixed gamma
    }
    
    # Different hyperparameters based on optimizer
    if args.optimizer == 'sgd':
        lr = trial.suggest_float('lr', 0.01, 1.0, log=True)
        optimizer_kwargs['momentum'] = trial.suggest_float('momentum', 0.0, 0.9)

    elif args.optimizer == 'adamw':
        lr = trial.suggest_float('lr', 1e-4, 1.0, log=True)
        beta1 = trial.suggest_float('beta1', 0.8, 0.95)
        beta2 = trial.suggest_float('beta2', 0.99, 0.999)
        optimizer_kwargs['betas'] = (beta1, beta2)
        optimizer_kwargs['weight_decay'] = 1e-2

    elif args.optimizer == 'fsgld':
        lr = trial.suggest_float('lr', 0.01, 1.0, log=True)
        if args.fixedbeta:
            beta_inv = args.betavalue
            trial.set_user_attr('beta_inv', beta_inv) # to store beta_inv to params list
        else:
            beta_inv = trial.suggest_float('beta_inv', args.betainvlow, args.betainvhigh, log=True)
        optimizer_kwargs['beta_inv'] = beta_inv
        if args.beta_coupling:
            # sigma will be zero if --rwp is on, so use beta_coupling only when --rwp is off.
            sigma = beta_inv ** ((1.0 + args.eta) / 4.0)
        else:
            sigma = trial.suggest_float('sigma', 0.001, 0.1, log=True)

        optimizer_kwargs['sigma'] = sigma
        optimizer_kwargs['n_pert'] = 1  # Fixed
        optimizer_kwargs['pert_type'] = 'normal'  # Fixed
        optimizer_kwargs['momentum'] = 0.0
        optimizer_kwargs['beta_coupling'] = args.beta_coupling
        optimizer_kwargs['eta'] = args.eta

    elif args.optimizer == 'sgld':
        lr = trial.suggest_float('lr', 0.01, 1.0, log=True)
        if args.fixedbeta:
            beta_inv = args.betavalue
            trial.set_user_attr('beta_inv', beta_inv) # to store beta_inv to params list
        else:
            beta_inv = trial.suggest_float('beta_inv', args.betainvlow, args.betainvhigh, log=True)
        optimizer_kwargs['beta_inv'] = beta_inv
        optimizer_kwargs['momentum'] = 0.0

    elif args.optimizer == 'sam':
        lr = trial.suggest_float('lr', 0.01, 1.0, log=True)
        optimizer_kwargs['rho'] = trial.suggest_float('rho', 1e-3, 1e-1, log=True)
        optimizer_kwargs['momentum'] = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer_kwargs['adaptive'] = False  
    elif args.optimizer == 'asam':
        lr = trial.suggest_float('lr', 0.01, 1.0, log=True)
        optimizer_kwargs['rho'] = trial.suggest_float('rho', 1e-3, 1e-1, log=True)
        optimizer_kwargs['momentum'] = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer_kwargs['adaptive'] = True       

    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, args.backbone, args.data_root)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    model = get_backbone(args.backbone, num_classes).to(device)
    
    # Create optimizer
    optimizer, scheduler = get_optimizer(model, args.optimizer, lr, **optimizer_kwargs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_test_acc = 0.0
    early_stop_patience = args.early_stop_patience
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.optimizer
        )
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        scheduler.step()
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Report intermediate value
        trial.report(test_acc, epoch)

        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            logger.info(f"Trial {trial.number} - Epoch {epoch}: "
                       f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    return best_test_acc


def run_optuna_optimization(args):
    """Run Optuna hyperparameter optimization"""
    
    # Create study
    study_name = f"{args.dataset}_{args.backbone}_{args.optimizer}_optuna_study"
    storage_name = f"sqlite:///{os.path.join(args.save_dir, study_name)}.db"
    
    if args.resume_study:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name
        )
        logger.info(f"Resuming study: {study_name}")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            load_if_exists=True
        )
        logger.info(f"Created new study: {study_name}")
        
    # Optimize 
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    n_completed = len(completed_trials)
    logger.info(f"Currently completed trials: {n_completed} / {args.n_trials}")

    remaining_needed = max(args.n_trials - n_completed, 0)

    if remaining_needed > 0:
        logger.info(f"Trying up to {remaining_needed} more trials in this run.")
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=remaining_needed,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=True
        )
    else:
        logger.info("Target number of COMPLETE trials already reached. Skipping optimization.")
    
    # Print statistics
    logger.info("\n" + "="*50)
    logger.info(f"Optimization completed for {args.optimizer}!")
    logger.info("="*50)
    
    # Best trial
    best_trial = study.best_trial
    logger.info(f"\nBest trial: {best_trial.number}")
    logger.info(f"Best value (test accuracy): {best_trial.value:.2f}%")
    logger.info(f"\nBest hyperparameters for {args.optimizer}:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Statistics
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    logger.info(f"\nStudy statistics:")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of completed trials: {len(completed_trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    
    # Save results
    results_file = os.path.join(args.save_dir, f"{study_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'optimizer': args.optimizer,
            'dataset': args.dataset,
            'backbone': args.backbone,
            'best_trial_number': best_trial.number,
            'best_value': best_trial.value,
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
            'n_completed': len(completed_trials),
            'n_pruned': len(pruned_trials),
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in study.trials
            ]
        }, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
        
    return study



def train_with_params(args, best_params, seed):
    """Train model with best hyperparameters found by Optuna"""
    
    logger.info("\n" + "="*50)
    logger.info(f"Training with best hyperparameters for {args.optimizer}")
    logger.info("="*50)
    
    # Extract parameters
    lr = best_params['lr']
    batch_size = 128  # Fixed
    weight_decay = 5e-4  # Fixed
    
    # Build optimizer kwargs
    optimizer_kwargs = {
        'weight_decay': weight_decay,
        'milestones': [50, 100],
        'gamma': 0.1
    }
    
    if args.optimizer == 'sgd':
        optimizer_kwargs['momentum'] = best_params.get('momentum', 0.9)
    
    elif args.optimizer == 'fsgld':
        if args.fixedbeta:
            beta_inv = args.betavalue
            # To contain beta_inv value inside the log, since it was out of optuna search.
            logger.info(f"beta_inv={beta_inv}")
        else:
            beta_inv = best_params.get('beta_inv', 1e-14)
        optimizer_kwargs['beta_inv'] = beta_inv
        if args.beta_coupling:
            sigma = beta_inv ** ((1.0 + args.eta) / 4.0)
        else:
            sigma = best_params.get('sigma', 0.001)
        optimizer_kwargs['sigma'] = sigma
        optimizer_kwargs['n_pert'] = 1 
        optimizer_kwargs['pert_type'] = args.pert_type if hasattr(args, 'pert_type') else 'normal'
        optimizer_kwargs['antithetic'] = args.antithetic if hasattr(args, 'antithetic') else False
        optimizer_kwargs['momentum'] = best_params.get('fsgld_momentum', 0.0)
        optimizer_kwargs['beta_coupling'] = args.beta_coupling if hasattr(args, 'beta_coupling') else False
        optimizer_kwargs['eta'] = args.eta
    
    elif args.optimizer == 'sgld':
        if args.fixedbeta:
            beta_inv = args.betavalue
            # To contain beta_inv value inside the log, since it was out of optuna search.
            logger.info(f"beta_inv={beta_inv}")
        else:
            beta_inv = best_params.get('beta_inv', 1e-14)
        optimizer_kwargs['beta_inv'] = beta_inv
        optimizer_kwargs['momentum'] = best_params.get('momentum', 0.0)

    elif args.optimizer == 'sam':
        optimizer_kwargs['rho'] = best_params.get('rho', 0.05)
        optimizer_kwargs['momentum'] = best_params.get('momentum', 0.9)
        optimizer_kwargs['adaptive'] = False  # Fixed
    
    # Set seed
    set_seed(seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, args.backbone, args.data_root
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    model = get_backbone(args.backbone, num_classes).to(device)
    
    # Create optimizer
    optimizer, scheduler = get_optimizer(model, args.optimizer, lr, **optimizer_kwargs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_test_acc = 0.0
    best_test_loss = float('inf')
    results = []
    
    for epoch in range(args.final_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.optimizer
        )
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        # Update best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best model
            save_path = os.path.join(args.save_dir, f'best_model_{args.optimizer}.pth')
            torch.save(model.state_dict(), save_path)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'time': epoch_time
        })
    
    # Final results
    final_test_acc = results[-1]['test_acc']
    last_n = min(10, len(results))
    avg_last_test_acc = np.mean([r['test_acc'] for r in results[-last_n:]])
    
    logger.info(f'=== FINAL RESULTS - SEED {seed} ===')
    logger.info(f'Best test accuracy: {best_test_acc:.2f}%')
    logger.info(f'Final test accuracy: {final_test_acc:.2f}%')
    logger.info(f'Average last {last_n} epochs: {avg_last_test_acc:.2f}%')
    
    final_model_path = os.path.join(args.save_dir, f'final_model_{args.optimizer}.pth')
    torch.save(model.state_dict(), final_model_path)
        
    return {
        'seed': seed,
        'best_test_acc': best_test_acc,
        'final_test_acc': final_test_acc,
        'avg_last_test_acc': avg_last_test_acc,
        'results': results
    }


def load_best_params(results_file):
    """Load best parameters from saved results file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results['best_params'], results['optimizer'], results['dataset'], results['backbone']


def train_multi_seeds(args):
    """Train with best params from saved results file using multiple seeds"""
    
    # Load best parameters
    if not os.path.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}")
    
    best_params, optimizer_name, dataset, backbone = load_best_params(args.results_file)
    
    logger.info("\n" + "="*60)
    logger.info(f"MULTI-SEED TRAINING WITH BEST HYPERPARAMETERS")
    logger.info("="*60)
    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Best params: {best_params}")
    logger.info("="*60)
    
    # Train with multiple seeds
    all_results = []
    
    for seed in args.seeds:
        logger.info(f"\n{'='*20} SEED {seed} {'='*20}")
        
        result = train_with_params(
            args, best_params, seed
        )
        all_results.append(result)
    
    # Aggregate results
    logger.info("\n" + "="*60)
    logger.info("MULTI-SEED RESULTS SUMMARY")
    logger.info("="*60)
    
    best_accs = [r['best_test_acc'] for r in all_results]
    final_accs = [r['final_test_acc'] for r in all_results]
    avg_accs = [r['avg_last_test_acc'] for r in all_results]
    
    for i, result in enumerate(all_results):
        logger.info(f"Seed {result['seed']:2d}: Best={result['best_test_acc']:6.2f}%, "
                   f"Final={result['final_test_acc']:6.2f}%, "
                   f"Avg={result['avg_last_test_acc']:6.2f}%")
    
    logger.info("-" * 60)
    logger.info(f"Best accuracy  - Mean: {np.mean(best_accs):6.2f}% ± {np.std(best_accs):5.2f}%")
    logger.info(f"Final accuracy - Mean: {np.mean(final_accs):6.2f}% ± {np.std(final_accs):5.2f}%")
    logger.info(f"Avg accuracy   - Mean: {np.mean(avg_accs):6.2f}% ± {np.std(avg_accs):5.2f}%")
    
    # Save multi-seed results
    multi_seed_results = {
        'optimizer': optimizer_name,
        'dataset': dataset,
        'backbone': backbone,
        'best_params': best_params,
        'seeds': args.seeds,
        'individual_results': all_results,
        'summary': {
            'best_acc_mean': float(np.mean(best_accs)),
            'best_acc_std': float(np.std(best_accs)),
            'final_acc_mean': float(np.mean(final_accs)),
            'final_acc_std': float(np.std(final_accs)),
            'avg_acc_mean': float(np.mean(avg_accs)),
            'avg_acc_std': float(np.std(avg_accs)),
        }
    }
    
    # Save results
    results_filename = f'multi_seed_results_{optimizer_name}_{dataset}_{backbone}.json'
    results_path = os.path.join(args.save_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(multi_seed_results, f, indent=2)
    
    logger.info(f"\nMulti-seed results saved to: {results_path}")
    
    return multi_seed_results




def train_with_best_params(args, best_params):
    """Train model with best hyperparameters found by Optuna"""
    
    logger.info("\n" + "="*50)
    logger.info(f"Training with best hyperparameters for {args.optimizer}")
    logger.info("="*50)
    
    # Extract parameters
    lr = best_params['lr']
    batch_size = 128  # Fixed
    weight_decay = 5e-4  # Fixed
    
    # Build optimizer kwargs
    optimizer_kwargs = {
        'weight_decay': weight_decay,
        'milestones': [50,100],
        'gamma': 0.1
    }
    
    if args.optimizer == 'sgd':
        optimizer_kwargs['momentum'] = best_params.get('momentum', 0.9)
    
    elif args.optimizer == 'fsgld':
        if args.fixedbeta:
            beta_inv = args.betavalue
            # To contain beta_inv value inside the log, since it was out of optuna search.
            logger.info(f"beta_inv={beta_inv}")
        else:
            beta_inv = best_params.get('beta_inv', 1e-14)
        optimizer_kwargs['beta_inv'] = beta_inv
        if args.beta_coupling:
            sigma = beta_inv ** ((1.0 + args.eta) / 4.0)
        else:
            sigma = best_params.get('sigma', 0.001)
        optimizer_kwargs['sigma'] = sigma
        optimizer_kwargs['n_pert'] = 1  
        optimizer_kwargs['pert_type'] = 'normal'  # Fixed
        optimizer_kwargs['momentum'] = best_params.get('fsgld_momentum', 0.0)
        optimizer_kwargs['beta_coupling'] = args.beta_coupling
        optimizer_kwargs['eta'] = args.eta
    
    elif args.optimizer == 'sgld':
        if args.fixedbeta:
            beta_inv = args.betavalue
            # To contain beta_inv value inside the log, since it was out of optuna search.
            logger.info(f"beta_inv={beta_inv}")
        else:
            beta_inv = best_params.get('beta_inv', 1e-14)
        optimizer_kwargs['beta_inv'] = beta_inv
        optimizer_kwargs['momentum'] = 0.0

    elif args.optimizer == 'sam':
        optimizer_kwargs['rho'] = best_params.get('rho', 0.05)
        optimizer_kwargs['momentum'] = best_params.get('momentum', 0.9)
        optimizer_kwargs['adaptive'] = False  # Fixed

    elif args.optimizer == 'asam':
        optimizer_kwargs['rho'] = best_params.get('rho', 0.05)
        optimizer_kwargs['momentum'] = best_params.get('momentum', 0.9)
        optimizer_kwargs['adaptive'] = True  # Fixed
        
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, args.backbone, args.data_root)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    model = get_backbone(args.backbone, num_classes).to(device)
    
    # Create optimizer
    optimizer, scheduler = get_optimizer(model, args.optimizer, lr, **optimizer_kwargs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_test_acc = 0.0
    best_test_loss = float('inf')
    results = []
    
    for epoch in range(args.final_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.optimizer
        )
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        # Update best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best model
            save_path = os.path.join(args.save_dir, f'best_model_{args.optimizer}.pth')
            torch.save(model.state_dict(), save_path)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'time': epoch_time
        })
    
    # Final results
    final_test_acc = results[-1]['test_acc']
    last_n = min(10, len(results))
    avg_last_test_acc = np.mean([r['test_acc'] for r in results[-last_n:]])
    
    logger.info(f'\n=== FINAL RESULTS WITH BEST PARAMS ({args.optimizer}) ===')
    logger.info(f'Best test accuracy: {best_test_acc:.2f}%')
    logger.info(f'Final test accuracy: {final_test_acc:.2f}%')
    logger.info(f'Average last {last_n} epochs: {avg_last_test_acc:.2f}%')
    
    # Save final results
    final_results_file = os.path.join(args.save_dir, f'final_training_results_{args.optimizer}.json')
    with open(final_results_file, 'w') as f:
        json.dump({
            'optimizer': args.optimizer,
            'dataset': args.dataset,
            'backbone': args.backbone,
            'best_params': best_params,
            'best_test_acc': best_test_acc,
            'final_test_acc': final_test_acc,
            'avg_last_test_acc': avg_last_test_acc,
            'results': results
        }, f, indent=2)
    
    logger.info(f'Results saved to {final_results_file}')
    
    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization for Noisy Label Training')
    
    # Model and dataset settings
    parser.add_argument('--backbone', type=str, default='resnet34', 
                       choices=['resnet34', 'resnet50','resnet34_cifar', 'resnet50_cifar','vit_b_16'], 
                       help='Backbone architecture')
    parser.add_argument('--dataset', type=str, default='cifar10N',
                       choices=['cifar10N', 'cifar100N', 'webvision'], 
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data', 
                       help='Data root directory')
    
    # Optimizer selection (IMPORTANT: this selects which optimizer to tune)
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'fsgld', 'sam', 'asam','sgld', 'adamw'], 
                       help='Optimizer to tune')
    parser.add_argument('--beta_coupling', action='store_true', help='Use coupled beta')
    parser.add_argument('--eta', type=float, default=0.1, help='for beta-sigma coupling, should use with beta_coupling on.')
    parser.add_argument('--betainvlow', type=float, default=1e-9, help='lower bound for sgld beta search range.')
    parser.add_argument('--betainvhigh', type=float, default=1e-7, help='upper bound for sgld beta search range.')
    parser.add_argument("--fixedbeta", action="store_true", help="Use a fixed beta_inv instead of Optuna-suggested beta_inv")
    parser.add_argument("--betavalue", type=float, default=1e-14, help="Fixed beta_inv value used when --fixedbeta is on")

    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=20, 
                       help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, 
                       help='Timeout for optimization in seconds')
    parser.add_argument('--n_jobs', type=int, default=1, 
                       help='Number of parallel jobs')
    parser.add_argument('--resume_study', action='store_true', 
                       help='Resume existing study')
    # Training settings
    parser.add_argument('--epochs', type=int, default=150, 
                       help='Number of epochs for each trial')
    parser.add_argument('--final_epochs', type=int, default=150, 
                       help='Number of epochs for final training')
    parser.add_argument('--early_stop_patience', type=int, default=150, 
                       help='Early stopping patience')
    parser.add_argument('--train_final', action='store_true', 
                       help='Train with best params after optimization')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=8, 
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./optuna_results', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info(f"Starting Optuna optimization for {args.optimizer} optimizer")
    logger.info(f"Dataset: {args.dataset}, Backbone: {args.backbone}")
    
    # Run Optuna optimization
    study = run_optuna_optimization(args)
    
    # Train with best hyperparameters
    if args.train_final:
        best_params = study.best_trial.params
        train_with_best_params(args, best_params)


if __name__ == '__main__':
    main()
