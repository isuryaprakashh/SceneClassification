import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

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


# -----------------------------------------------------------
# TRAIN ONE EPOCH
# -----------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
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


# -----------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# SAVE CHECKPOINT
# -----------------------------------------------------------
def save_checkpoint(model_state, classes, image_size, model_name):
    model_dir = ARTIFACT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model_state,
        "arch": model_name,
        "image_size": image_size,
    }, model_dir / "best_model.pt")

    with open(model_dir / "classes.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    print(f"✔ Saved best model for {model_name}!")


# -----------------------------------------------------------
# TRAIN SINGLE MODEL
# -----------------------------------------------------------
def train_model(model_name, ModelClass, num_classes, train_loader, val_loader, device, classes):
    print("\n============================================")
    print(f"🔵 Training Model → {model_name}")
    print("============================================")

    model = ModelClass(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    best_acc = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp
        )
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        print(f"[{model_name}] Epoch {epoch}/{EPOCHS} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model.state_dict(), classes, IMAGE_SIZE, model_name)
            print(f"🔥 Best {model_name} Accuracy Updated: {best_acc:.2f}%")

    return best_acc


# -----------------------------------------------------------
# TRAIN MODEL WITH DETAILED LOGGING
# -----------------------------------------------------------
def train_model_with_history(
    model_name, ModelClass, num_classes, train_loader, val_loader, device, classes
):
    """
    Train a model while recording detailed epoch-wise metrics.
    
    Returns:
        best_acc: Best validation accuracy achieved
        history: Dict containing training/validation metrics per epoch
        training_time: Total training time in seconds
    """
    print("\n" + "=" * 60)
    print(f"🔵 Training Model → {model_name}")
    print("=" * 60)

    model = ModelClass(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': [],
    }
    
    best_acc = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_amp=use_amp
        )
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        print(f"[{model_name}] Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model.state_dict(), classes, IMAGE_SIZE, model_name)
            print(f"🔥 Best {model_name} Accuracy Updated: {best_acc:.2f}%")

    training_time = time.time() - start_time
    print(f"\n✔ {model_name} training completed in {training_time:.1f}s")
    
    return best_acc, history, training_time


# -----------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("SCENE CLASSIFICATION - MODEL TRAINING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device → {device}")

    train_loader, val_loader, classes = pre.load_data(
        DATA_DIR, batch_size=32, augment=True,
        num_workers=min(8, os.cpu_count())
    )

    num_classes = len(classes)
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Training samples: {len(train_loader.dataset)}")
    print(f"[INFO] Validation samples: {len(val_loader.dataset)}")

    MODELS = {
        "ANNClassifier": models.ANNClassifier,
        "CNNRegularized": models.CNNRegularized,
        "ResNetTransfer": models.ResNetTransfer
    }

    results = {}
    all_histories = {}
    training_times = {}

    # Train all models
    for model_name, ModelClass in MODELS.items():
        acc, history, train_time = train_model_with_history(
            model_name, ModelClass, num_classes,
            train_loader, val_loader, device, classes
        )
        results[model_name] = acc
        all_histories[model_name] = history
        training_times[model_name] = train_time

    # Save training history for visualization
    history_path = ARTIFACT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"\n✔ Training history saved to: {history_path}")

    # Show final comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Best Val Acc':>12} {'Train Time':>12}")
    print("-" * 46)
    for name in MODELS.keys():
        acc = results[name]
        t = training_times[name]
        print(f"{name:<20} {acc:>11.2f}% {t:>11.1f}s")

    best_model = max(results, key=results.get)
    print(f"\n🏆 BEST MODEL: {best_model} → {results[best_model]:.2f}%")
    
    # Save comprehensive results
    summary = {
        "training_date": datetime.now().isoformat(),
        "device": str(device),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "image_size": IMAGE_SIZE,
        "num_classes": num_classes,
        "classes": classes,
        "results": {
            name: {
                "best_val_accuracy": results[name],
                "training_time_seconds": training_times[name],
                "final_train_acc": all_histories[name]['train_acc'][-1],
                "final_val_acc": all_histories[name]['val_acc'][-1],
                "final_train_loss": all_histories[name]['train_loss'][-1],
                "final_val_loss": all_histories[name]['val_loss'][-1],
            }
            for name in MODELS.keys()
        },
        "best_model": best_model,
    }
    
    summary_path = ARTIFACT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✔ Training summary saved to: {summary_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
