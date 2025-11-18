from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformSubset(Dataset):
    """Apply a transform lazily to a subset produced by random_split."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, target = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target


def _build_transforms(augment: bool, image_size: int = 224):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if not augment:
        return transforms.Compose(base)

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_data(
    data_dir: str | Path,
    batch_size: int = 32,
    augment: bool = True,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    max_images_per_class: int | None = None,
) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    """
    Load dataset using ImageFolder, split into train/validation, and
    return dataloaders plus class names.
    
    Args:
        max_images_per_class: If set, limit the number of images per class (for faster training).
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    base_dataset = datasets.ImageFolder(root=root)
    classes = base_dataset.classes  # Save classes before creating subset
    
    # Limit images per class if specified
    if max_images_per_class is not None and max_images_per_class > 0:
        import random
        random.seed(seed)
        # Use dataset's samples attribute (path, class_index) without loading images
        indices_by_class = {}
        for idx, (path, label) in enumerate(base_dataset.samples):
            if label not in indices_by_class:
                indices_by_class[label] = []
            indices_by_class[label].append(idx)
        
        # Sample max_images_per_class from each class
        selected_indices = []
        for label, indices in indices_by_class.items():
            if len(indices) > max_images_per_class:
                selected_indices.extend(random.sample(indices, max_images_per_class))
            else:
                selected_indices.extend(indices)
        
        # Shuffle selected indices for better training
        random.shuffle(selected_indices)
        
        # Create a subset with selected indices
        from torch.utils.data import Subset
        base_dataset = Subset(base_dataset, selected_indices)
    
    total = len(base_dataset)
    if total == 0:
        raise RuntimeError(f"No images found under {root}")

    val_size = max(1, int(total * val_split))
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    train_transform = _build_transforms(augment=True and augment)
    val_transform = _build_transforms(augment=False)

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )

    return train_loader, val_loader, classes


def visualize_data(train_loader: DataLoader, classes: Sequence[str], num_samples: int = 5):
    """
    Display a handful of images from the first training batch.
    Safe-guards for headless environments.
    """
    if train_loader is None or len(train_loader) == 0:
        print("Train loader empty, skipping visualization.")
        return

    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        print("No data available for visualization.")
        return

    images = images[:num_samples].cpu().numpy().transpose(0, 2, 3, 1)
    images = (images * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    images = np.clip(images, 0, 1)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for idx in range(num_samples):
        axes[idx].imshow(images[idx])
        axes[idx].axis("off")
        axes[idx].set_title(f"Label: {classes[labels[idx].item()]}")
    plt.tight_layout()
    plt.show()

