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
from typing import Dict, Any
import json
import logging

from optimizers import *
from dataloader import get_dataset

from models import *

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
        weights = ViT_B_16_Weights.DEFAULT  # or None if you don't want pretrained
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
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
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs.get('milestones',[50,100]), gamma=0.1)
    
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=kwargs.get('weight_decay', 5e-4)
        )
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs.get('milestones',[50, 100]), gamma=0.1)


    elif optimizer_name == 'sgld':
        optimizer = SGLD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.0),
            weight_decay=kwargs.get('weight_decay', 5e-4),
            beta_inv=kwargs.get('beta_inv', 1e-14)
        )
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs.get('milestones',[50, 100]), gamma=0.1)


    elif optimizer_name == 'fSGLD':
        optimizer = fSGLD(
            model.parameters(),
            lr=lr,
            sigma=kwargs.get('sigma', 0.001),
            n_pert=kwargs.get('n_pert', 1),
            momentum=kwargs.get('momentum', 0.0),
            weight_decay=kwargs.get('weight_decay', 5e-4),
            beta_inv=kwargs.get('beta_inv', 1e-14),
            pert_type=kwargs.get('pert_type', 'normal')
        )
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs.get('milestones',[50, 100]), gamma=0.1)

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
        scheduler = optim.lr_scheduler.MultiStepLR(base_opt, milestones=kwargs.get('milestones',[50, 100]), gamma=0.1)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch_data in enumerate(train_loader):

        # Handle different data formats
        if len(batch_data) == 2:
            data, target = batch_data
        elif len(batch_data) == 3:
            data, target, _ = batch_data  # ignore clean labels during training
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        data, target = data.to(device), target.to(device)
        
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss, output
        
        if args.optimizer in ['fsgld', 'sgld', 'sam']:
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
        
        if batch_idx % 100 == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                       f'Acc: {100.*correct/total:.2f}%')
    
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
                data, target, _ = batch_data  # ignore additional info
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


def main():
    parser = argparse.ArgumentParser(description='Noisy Label Training')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='resnet34', 
                       choices=['resnet34', 'resnet50','resnet34_cifar', 'resnet50_cifar', 'vit'], help='Backbone architecture')

    parser.add_argument('--dataset', type=str, default='cifar10N',
                       choices=['cifar10N', 'cifar100N', 'webvision'], 
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')

    # Training settings
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'fsgld', 'sam'], help='Optimizer')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 100], 
                       help='Learning rate decay milestones (list of epochs)')
    
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # fSGLD specific
    parser.add_argument('--sigma', type=float, default=0.1, help='fSGLD perturbation scale')
    parser.add_argument('--n_pert', type=int, default=1, help='fSGLD number of perturbations')
    parser.add_argument('--beta_inv', type=float, default=1e-14, help='fSGLD Langevin noise scale')
    parser.add_argument('--pert_type', type=str, default='normal',
                       choices=['normal', 'antithetic'], help='fSGLD perturbation type')
    
    # SAM specific
    parser.add_argument('--rho', type=float, default=0.05, help='SAM perturbation radius')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive SAM')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory with experiment name
    exp_name = f"{args.dataset}_{args.backbone}_{args.optimizer}"
    if args.optimizer == 'fsgld':
        exp_name += f"_sigma{args.sigma}_seed{args.seed}"
    elif args.optimizer == 'sam':
        exp_name += f"_rho{args.rho}"
        if args.adaptive:
            exp_name += "_adaptive"
    exp_name += f"_lr{args.lr}_bs{args.batch_size}_seed{args.seed}"
    
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Results will be saved to: {save_dir}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
   # Load dataset
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, args.backbone, args.data_root
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f"Dataset: {args.dataset}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = get_backbone(args.backbone, num_classes).to(device)
    logger.info(f"Model: {args.backbone}, Classes: {num_classes}")
    
    # Create optimizer
    optimizer_kwargs = {
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'sigma': args.sigma,
        'n_pert': args.n_pert,
        'beta_inv': args.beta_inv,
        'pert_type': args.pert_type,
        'rho': args.rho,
        'adaptive': args.adaptive
    }
    
    optimizer, scheduler = get_optimizer(model, args.optimizer, args.lr, **optimizer_kwargs)
    logger.info(f"Optimizer: {args.optimizer}")
    

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_test_acc = 0.0
    best_test_loss = float('inf')
    best_acc_epoch = 0
    best_loss_epoch = 0
    results = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        

        scheduler.step()


        epoch_time = time.time() - start_time
        
        logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
        
        # Save best models based on different criteria
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_acc_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_acc_model.pth'))

        # Record results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'time': epoch_time
        })
    
    # Calculate final metrics
    final_test_acc = results[-1]['test_acc']
    final_test_loss = results[-1]['test_loss']
    
    # Average of last 10 epochs (or available epochs if less than 10)
    last_n = min(10, len(results))
    avg_last_test_acc = np.mean([r['test_acc'] for r in results[-last_n:]])
    avg_last_test_loss = np.mean([r['test_loss'] for r in results[-last_n:]])
    
    logger.info(f'=== FINAL RESULTS ===')
    logger.info(f'Best test accuracy: {best_test_acc:.2f}% (epoch {best_acc_epoch})')
    logger.info(f'Best test loss: {best_test_loss:.4f} (epoch {best_loss_epoch})')
    logger.info(f'Final test accuracy: {final_test_acc:.2f}%')
    logger.info(f'Final test loss: {final_test_loss:.4f}')
    logger.info(f'Average last {last_n} epochs - Acc: {avg_last_test_acc:.2f}%, Loss: {avg_last_test_loss:.4f}')
    
    # Save results
    results_file = os.path.join(save_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_name': exp_name,
            'args': vars(args),
            'best_test_acc': best_test_acc,
            'best_test_loss': best_test_loss,
            'best_acc_epoch': best_acc_epoch,
            'best_loss_epoch': best_loss_epoch,
            'final_test_acc': final_test_acc,
            'final_test_loss': final_test_loss,
            'avg_last_test_acc': avg_last_test_acc,
            'avg_last_test_loss': avg_last_test_loss,
            'results': results
        }, f, indent=2)
    
    logger.info(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()