import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler  # new AMP API

import Preprocess as pre
import models1 as models

torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)

IMAGE_SIZE = 224
DATA_DIR = Path("DATASET") / "images"
ARTIFACT_DIR = Path("artifacts")

EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
USE_TRANSFER_LEARNING = True


def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None and device.type == "cuda":
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total_correct += pred.eq(labels).sum().item()
        total_items += labels.size(0)

    return total_loss / len(train_loader), (total_correct / total_items) * 100


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100, loss_sum / len(loader)


def save_checkpoint(model_state, classes, image_size):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model_state,
        "arch": "resnet_transfer",
        "image_size": image_size,
    }, ARTIFACT_DIR / "best_model.pt")

    with open(ARTIFACT_DIR / "classes.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    print("✔ Saved best model!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on → {device}")

    train_loader, val_loader, classes = pre.load_data(
        DATA_DIR,
        batch_size=32,
        augment=True,
        num_workers=min(8, os.cpu_count()),
        max_images_per_class=None,
    )

    print(f"Classes: {classes}")

    num_classes = len(classes)

    if USE_TRANSFER_LEARNING:
        model = models.ResNetTransfer(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
    else:
        model = models.CNNRegularized(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)

    # ---------------------- FIXED AMP SCALER ----------------------
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    # --------------------------------------------------------------

    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler, use_amp=use_amp
        )
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model.state_dict(), classes, IMAGE_SIZE)

    print(f"\n🎉 Final Best Accuracy: {best_acc:.2f}%\n")


if __name__ == "__main__":
    main()
