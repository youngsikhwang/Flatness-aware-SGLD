import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
import sys

logger = logging.getLogger(__name__)


class NoisyDataset(Dataset):
    """Generic noisy dataset loader for real-world noisy datasets"""
    
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to the dataset file (e.g., .npz, .pkl, or directory)
            transform: Transform to apply to images
        """
        self.transform = transform
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load data from file - customize based on your dataset format"""
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            self.images = data['images']
            self.labels = data['labels']
        elif data_path.endswith('.pkl'):
            import pickle
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.images = data['images']
            self.labels = data['labels']
        else:
            # Assume it's a directory with subdirectories for each class
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(data_path)
            self.images = [dataset[i][0] for i in range(len(dataset))]
            self.labels = [dataset[i][1] for i in range(len(dataset))]
        
        logger.info(f"Loaded {len(self.images)} samples from {data_path}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            from PIL import Image
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)  # Convert to HWC
            image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



# Import existing CIFAR dataset classes if available
try:
    from data.cifar import CIFAR10, CIFAR100
    CIFAR_AVAILABLE = True
except ImportError:
    logger.warning(f"Current working directory: {os.getcwd()}")
    logger.warning(f"Python path: {sys.path}")
    logger.warning(f"__file__ location: {__file__ if '__file__' in globals() else 'Not available'}")

    # Check if data directory exists
    data_dir = os.path.join(os.getcwd(), 'data')
    logger.warning(f"Data directory exists: {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        logger.warning(f"Data directory contents: {os.listdir(data_dir)}")

    # Check if cifar.py exists
    cifar_file = os.path.join(data_dir, 'cifar.py')
    logger.warning(f"cifar.py exists: {os.path.exists(cifar_file)}")


    CIFAR_AVAILABLE = False
    logger.warning("CIFAR dataset classes not found, using fallback NoisyDataset")


def get_transforms(dataset_name):
    """Get appropriate transforms for each dataset"""
    
    if dataset_name == 'cifar10N':
        # CIFAR-10 normalization
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    elif dataset_name == 'cifar100N':
        # CIFAR-100 normalization
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    else:
        # Default CIFAR-10 normalization for other datasets
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    return train_transform, test_transform


def get_dataset(dataset_name: str, data_root: str, train_file: str = None, test_file: str = None, 
                noise_type: str = None, is_human: bool = True):
    """Get dataset with transforms"""
    
    train_transform, test_transform = get_transforms(dataset_name)
    
    # Use existing CIFAR classes if available
    if CIFAR_AVAILABLE and dataset_name in ['cifar10N', 'cifar100N']:
        if dataset_name == 'cifar10N':
            noise_file = os.path.join(data_root, 'CIFAR-10_human.pt')
            train_dataset = CIFAR10(
                root=data_root,
                download=True,
                train=True,
                transform=train_transform,
                noise_type='aggre_label',
                noise_path=noise_file,
                is_human=True
            )
            test_dataset = CIFAR10(
                root=data_root,
                download=True,
                train=False,
                transform=test_transform,
                noise_type='clean'
            )
            num_classes = 10
            
        elif dataset_name == 'cifar100N':
            noise_file = os.path.join(data_root, 'CIFAR-100_human.pt')
            train_dataset = CIFAR100(
                root=data_root,
                download=True,
                train=True,
                transform=train_transform,
                noise_type='noisy_label',
                noise_path=noise_file,
                is_human=True
            )
            test_dataset = CIFAR100(
                root=data_root,
                download=True,
                train=False,
                transform=test_transform,
                noise_type='clean'
            )
            num_classes = 100
    
    else:
        # Fallback to generic dataset loader
        # Set default file names if not provided
        if train_file is None:
            train_file = f"{dataset_name}_train.npz"
        if test_file is None:
            test_file = f"{dataset_name}_test.npz"
        
        # Load datasets
        if dataset_name == 'cifar10N':
            train_dataset = NoisyDataset(
                os.path.join(data_root, train_file), 
                transform=train_transform
            )
            test_dataset = NoisyDataset(
                os.path.join(data_root, test_file),
                transform=test_transform
            )
            num_classes = 10
            
        elif dataset_name == 'cifar100N':
            train_dataset = NoisyDataset(
                os.path.join(data_root, train_file),
                transform=train_transform
            )
            test_dataset = NoisyDataset(
                os.path.join(data_root, test_file),
                transform=test_transform
            )
            num_classes = 100
            
        elif dataset_name == 'animal10N':
            train_dataset = NoisyDataset(
                os.path.join(data_root, train_file),
                transform=train_transform
            )
            test_dataset = NoisyDataset(
                os.path.join(data_root, test_file),
                transform=test_transform
            )
            num_classes = 10
            
        elif dataset_name == 'clothing1M':
            train_dataset = NoisyDataset(
                os.path.join(data_root, train_file),
                transform=train_transform
            )
            test_dataset = NoisyDataset(
                os.path.join(data_root, test_file),
                transform=test_transform
            )
            num_classes = 14
            
        elif dataset_name == 'webvision':
            train_dataset = NoisyDataset(
                os.path.join(data_root, train_file),
                transform=train_transform
            )
            test_dataset = NoisyDataset(
                os.path.join(data_root, test_file),
                transform=test_transform
            )
            num_classes = 50
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes