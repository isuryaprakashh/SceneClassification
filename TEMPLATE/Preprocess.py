import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
def load_data(data_dir, batch_size=32, augment=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = datasets.ImageFolder(root=f"{data_dir}", transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    #train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    #val_data = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes

def visualize_data(train_loader, classes, num_samples=5):
    # Get a batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    images = images.numpy().transpose(0, 2, 3, 1)  # Convert CHW to HWC for matplotlib
   # images = (images * 0.5) + 0.5  # De-normalize (if normalization was applied)

# Display the batch
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        #axes[i].set_title(f"Label: {labels[i]}")
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
    plt.show()

