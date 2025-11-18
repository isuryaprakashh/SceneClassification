import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import Preprocess as pre
import models1 as models

IMAGE_SIZE = 224
DATA_DIR = Path("DATASET") / "images"
ARTIFACT_DIR = Path("artifacts")
EPOCHS = 5  # Reduced from 15 for faster training
LR = 1e-3  # Increased learning rate for faster convergence
WEIGHT_DECAY = 1e-5  # Reduced weight decay to allow model to learn more
USE_TRANSFER_LEARNING = True  # Use ResNet transfer learning for much better accuracy


def train(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss


def evaluate_random_minibatch(model, val_loader, criterion, device, num_batches=1):
    model.eval()
    batches = list(val_loader)
    if not batches:
        print("Validation loader empty, skipping random mini-batch evaluation.")
        return None, None
    num_batches = min(num_batches, len(batches))
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(num_batches):
            images, labels = batches[i]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_accuracy = 100 * correct / total
    avg_loss = val_loss / num_batches
    print(f"Random Mini-Batch Eval → Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
    return avg_accuracy, avg_loss


def save_checkpoint(model_state: dict, classes, image_size: int, arch: str = "cnn_regularized"):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model_state,
            "arch": arch,
            "image_size": image_size,
        },
        ARTIFACT_DIR / "best_model.pt",
    )
    with (ARTIFACT_DIR / "classes.json").open("w", encoding="utf-8") as fp:
        json.dump({"classes": classes}, fp, indent=2)
    print(f"Artifacts stored under {ARTIFACT_DIR.resolve()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = pre.load_data(
        data_dir=DATA_DIR,
        batch_size=32,
        augment=True,
        val_split=0.2,
        num_workers=0,
        max_images_per_class=10,  # Limit to 10 images per class for faster training
    )

    print(f"\nNumber of classes: {len(classes)}")
    print(f"Class names: {classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("\nVisualizing a few training images...")
    pre.visualize_data(train_loader, classes)

    num_classes = len(classes)
    
    # Use transfer learning for much better accuracy with limited data
    if USE_TRANSFER_LEARNING:
        print("\nUsing ResNet Transfer Learning (much better for scene classification!)")
        model = models.ResNetTransfer(num_classes=num_classes, input_size=IMAGE_SIZE, dropout=0.3).to(device)
        # Lower learning rate for transfer learning (pretrained weights)
        optimizer = optim.Adam(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
    else:
        print("\nUsing CNNRegularized model...")
        model = models.CNNRegularized(num_classes=num_classes, input_size=IMAGE_SIZE).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"\nTraining the model ({model.__class__.__name__})...")
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, device, epochs=1)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.2f}%")

    print(f"\nFinal best validation accuracy: {best_val_acc:.2f}%")
    evaluate_random_minibatch(model, val_loader, criterion, device, num_batches=1)

    arch_name = "resnet_transfer" if USE_TRANSFER_LEARNING else "cnn_regularized"
    save_checkpoint(model.state_dict(), classes, IMAGE_SIZE, arch=arch_name)


if __name__ == "__main__":
    main()
