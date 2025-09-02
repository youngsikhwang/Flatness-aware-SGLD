import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models import ViT_B_16_Weights
from torchvision.datasets import ImageFolder
import logging
import sys

from data.CIFAR.cifar import CIFAR10, CIFAR100

logger = logging.getLogger(__name__)


class NoisyDataset(Dataset):
    """Generic noisy dataset loader for real-world noisy datasets (npz/pkl/dir).
       - npz/pkl: memory load
       - dir(ImageFolder): lazy-load
    """
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self._is_imagefolder = False
        self.load_data(data_path)

    def load_data(self, data_path):
        if isinstance(data_path, tuple) and len(data_path) == 2:
            data_path, self._is_imagefolder = data_path

        if data_path.endswith('.npz'):
            data = np.load(data_path)
            self.images = data['images']
            self.labels = data['labels']
            self._is_imagefolder = False
        elif data_path.endswith('.pkl'):
            import pickle
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.images = data['images']
            self.labels = data['labels']
            self._is_imagefolder = False
        else:
            # Directory: lazy-load by ImageFolder
            self.dataset = ImageFolder(data_path)
            self._is_imagefolder = True

        if self._is_imagefolder:
            logger.info(f"[ImageFolder] {len(self.dataset)} samples from {data_path}")
        else:
            logger.info(f"[Array] {len(self.images)} samples from {data_path}")

    def __len__(self):
        if self._is_imagefolder:
            return len(self.dataset)
        return len(self.labels)

    def __getitem__(self, idx):
        if self._is_imagefolder:
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, np.ndarray):
            from PIL import Image
            # CHW -> HWC
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = image.transpose(1, 2, 0)
            image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        return image, label


class NoisyImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.ds = ImageFolder(root=root, transform=transform)
        self.transform = transform
        self.classes = self.ds.classes
        self.class_to_idx = self.ds.class_to_idx
        self.samples = self.ds.samples
        self.imgs = self.ds.imgs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

def get_transforms(dataset_name, model_name):
    """Get appropriate transforms for each dataset."""
    if str(model_name).lower().startswith("vit"):
        weights = ViT_B_16_Weights.DEFAULT
        train_transform = weights.transforms(antialias=True)
        test_transform  = weights.transforms(antialias=True)
        return train_transform, test_transform

    if dataset_name == 'webvision':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif dataset_name == 'cifar10N':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset_name == 'cifar100N':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    return train_transform, test_transform


def get_dataset(dataset_name: str,
                model_name,
                data_root: str):
    """Get dataset with transforms.
       - webvision: ImageFolder( data_root/train , data_root/val )
       - cifarN: use CIFAR class if exists, otherwise npz/pkl/dir fallback
    """
    train_transform, test_transform = get_transforms(dataset_name, model_name)

    if dataset_name == 'cifar10N':
        noise_file = os.path.join(data_root, 'CIFAR/CIFAR-10_human.pt')
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
        return train_dataset, test_dataset, num_classes

    elif dataset_name == 'cifar100N':
        noise_file = os.path.join(data_root, 'CIFAR/CIFAR-100_human.pt')
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

        return train_dataset, test_dataset, num_classes

    elif dataset_name == 'webvision':
        train_dir = os.path.join(data_root, 'WebVision/train')
        val_dir   = os.path.join(data_root, 'WebVision/val')

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"[webvision] train dir not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"[webvision] val dir not found: {val_dir}")

        train_dataset = NoisyImageFolder(train_dir, transform=train_transform)
        test_dataset  = NoisyImageFolder(val_dir,   transform=test_transform)
        num_classes = len(train_dataset.classes)

        logger.info(f"[webvision] classes: {num_classes}")
        logger.info(f"[webvision] train samples: {len(train_dataset)}, val samples: {len(test_dataset)}")
        return train_dataset, test_dataset, num_classes
