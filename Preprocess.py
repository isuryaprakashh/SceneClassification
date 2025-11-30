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
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label


def _build_transforms(augment=True, image_size=224):
    if not augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_data(
    data_dir,
    batch_size=32,
    augment=True,
    val_split=0.2,
    seed=42,
    num_workers=4,
    max_images_per_class=None
):
    root = Path(data_dir)
    dataset = datasets.ImageFolder(root=root)
    classes = dataset.classes

    if max_images_per_class:
        import random
        random.seed(seed)
        idx_map = {}
        for i, (_, label) in enumerate(dataset.samples):
            idx_map.setdefault(label, []).append(i)

        selected = []
        for label, idxs in idx_map.items():
            if len(idxs) > max_images_per_class:
                selected.extend(random.sample(idxs, max_images_per_class))
            else:
                selected.extend(idxs)

        dataset = Subset(dataset, selected)

    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_transform = _build_transforms(True)
    val_transform = _build_transforms(False)

    train_ds = TransformSubset(train_subset, train_transform)
    val_ds = TransformSubset(val_subset, val_transform)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return train_loader, val_loader, classes


def visualize_data(train_loader, classes, num_samples=5):
    try:
        images, labels = next(iter(train_loader))
    except:
        return

    images = images[:num_samples].permute(0, 2, 3, 1).numpy()
    images = images * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    images = np.clip(images, 0, 1)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")
    plt.show()
