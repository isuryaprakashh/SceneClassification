# Scene Classification using Deep Learning

A comprehensive scene classification system using multiple deep learning architectures including ANN, CNN, and ResNet-18 transfer learning.

## Project Overview

This project implements and compares three neural network architectures for classifying scene images into 47 distinct categories. The system includes training pipelines, comprehensive evaluation metrics, and a Flask-based web inference interface.

## Dataset

- **Categories**: 47 scene classes (beaches, classrooms, cities, forests, etc.)
- **Image Format**: RGB images of varying sizes
- **Preprocessing**: Resized to 224×224 pixels, normalized using ImageNet statistics

> **Data Split**: 80% training / 20% validation (random split with seed=42)

## Methodology

### 1. Data Preprocessing Pipeline

```
Raw Images → Resize (256×256) → Random Crop (224×224) → Horizontal Flip 
           → Color Jitter → ToTensor → Normalize (ImageNet mean/std)
```

| Transform | Training | Validation |
|-----------|----------|------------|
| Resize | 256×256 | 224×224 |
| RandomResizedCrop | 224×224 (scale 0.8-1.0) | — |
| RandomHorizontalFlip | p=0.5 | — |
| ColorJitter | brightness/contrast/saturation=0.2 | — |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Same |

### 2. Model Architectures

#### ANN Classifier (Baseline)
- **Architecture**: 4-layer fully connected network
- **Structure**: Flatten → FC(1024) → FC(512) → FC(256) → FC(num_classes)
- **Regularization**: Dropout (p=0.3) after each hidden layer
- **Parameters**: ~154M parameters

#### CNN Regularized
- **Architecture**: 4 convolutional blocks + 3 FC layers
- **Structure**: 
  ```
  Conv(3→32) → BN → ReLU → MaxPool(2) →
  Conv(32→64) → BN → ReLU → MaxPool(2) →
  Conv(64→128) → BN → ReLU → MaxPool(2) →
  Conv(128→256) → BN → ReLU → MaxPool(2) →
  Flatten → FC(512) → FC(256) → FC(num_classes)
  ```
- **Regularization**: Batch Normalization + Dropout (p=0.25)
- **Parameters**: ~15M parameters

#### ResNet-18 Transfer Learning
- **Base Model**: ResNet-18 pretrained on ImageNet
- **Modification**: Custom classifier head
  ```
  ResNet Features → Dropout(0.3) → FC(512) → ReLU → 
                    Dropout(0.3) → FC(num_classes)
  ```
- **Strategy**: Fine-tune all layers
- **Parameters**: ~11.7M parameters

### 3. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Batch Size | 32 |
| Epochs | 5 |
| LR Scheduler | ReduceLROnPlateau (patience=2) |
| Loss Function | CrossEntropyLoss |
| Mixed Precision | Enabled (CUDA only) |

### 4. Evaluation Metrics

The evaluation framework provides:
- **Classification Accuracy**: Top-1, Top-3, Top-5
- **Per-class Metrics**: Precision, Recall, F1-Score
- **Macro/Weighted Averages**: Aggregated F1 and precision/recall
- **ROC Curves & AUC**: One-vs-Rest multi-class ROC analysis
- **Confusion Matrix**: Normalized class-wise prediction patterns

## Project Structure

```
SceneClassification/
├── main.py              # Model training script
├── evaluate.py          # Comprehensive evaluation
├── visualizations.py    # Chart generation utilities
├── models1.py           # Model architecture definitions
├── Preprocess.py        # Data loading and augmentation
├── app.py               # Flask web inference app
├── requirements.txt     # Python dependencies
├── artifacts/           # Trained models and results
│   ├── best_model.pt
│   ├── classes.json
│   ├── training_history.json
│   ├── training_summary.json
│   ├── evaluation_results.json
│   ├── confusion_matrix.png
│   └── roc_curves.png
└── DATASET/images/      # Scene image dataset
```

## How to Reproduce

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python main.py
```

This trains all three architectures and saves:
- Best model checkpoints for each architecture
- `training_history.json` — epoch-wise metrics
- `training_summary.json` — final comparison summary

### 3. Evaluate Models

```bash
python evaluate.py --model_dir artifacts --data_dir DATASET/images
```

Generates:
- `evaluation_results.json` — comprehensive metrics
- `confusion_matrix.png` — visual confusion matrix
- `roc_curves.png` — ROC curves with AUC

### 4. Generate Visualizations

```bash
python visualizations.py --results_dir artifacts
```

Creates publication-ready charts for training curves and model comparison.

### 5. Run Web Interface

```bash
python app.py
```

Access at `http://localhost:5000` to upload and classify scene images.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- Flask
- scikit-learn
- matplotlib
- seaborn
- Pillow
- numpy

## Hardware Used

- GPU: NVIDIA CUDA-compatible (optional, CPU supported)
- RAM: 8GB minimum recommended
- Storage: ~500MB for models and dataset

## References

- ResNet: He et al., "Deep Residual Learning for Image Recognition" (2016)
- ImageNet Statistics: Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database" (2009)
- Adam Optimizer: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
