from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as torchvision_models


def _infer_flatten_dim(conv_layers: nn.Sequential, input_size: int = 224) -> int:
    with torch.no_grad():
        sample = torch.zeros(1, 3, input_size, input_size)
        features = conv_layers(sample)
    return features.view(1, -1).shape[1]


# --------------------------------------------------------------------
#  ANN CLASSIFIER  (Fully Connected Neural Network)
# --------------------------------------------------------------------
class ANNClassifier(nn.Module):
    """
    Simple ANN (MLP) replacing CNN with fully-connected layers only.
    Input: 224x224 RGB images → flatten → linear layers.
    """

    def __init__(self, num_classes: int, input_size: int = 224):
        super().__init__()

        self.flatten_dim = 3 * input_size * input_size  # 3×224×224 = 150,528

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------
#  CNN REGULARIZED
# --------------------------------------------------------------------
class CNNRegularized(nn.Module):
    """Simple CNN with batchnorm + dropout."""

    def __init__(self, num_classes: int, input_size: int = 224, dropout=0.25):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flattened_dim = _infer_flatten_dim(self.conv_layers, input_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))


# --------------------------------------------------------------------
#  RESNET TRANSFER LEARNING
# --------------------------------------------------------------------
class ResNetTransfer(nn.Module):
    """Transfer learning ResNet18 pretrained."""

    def __init__(self, num_classes: int, input_size=224, dropout=0.3, pretrained=True):
        super().__init__()

        try:
            resnet = torchvision_models.resnet18(
                weights=torchvision_models.ResNet18_Weights.DEFAULT
            )
        except:
            resnet = torchvision_models.resnet18(pretrained=True)

        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self.model = resnet

    def forward(self, x):
        return self.model(x)


__all__ = [
    "ANNClassifier",
    "CNNRegularized",
    "ResNetTransfer",
]
