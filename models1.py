import torch
import torch.nn as nn
from torchvision import models as torchvision_models


def _infer_flatten_dim(conv_layers: nn.Sequential, input_size: int = 224) -> int:
    with torch.no_grad():
        sample = torch.zeros(1, 3, input_size, input_size)
        features = conv_layers(sample)
    return features.view(1, -1).shape[1]


class CNNRegularized(nn.Module):
    """
    CNN classifier from the faculty template with batch normalization and dropout.
    """

    def __init__(self, num_classes: int, input_size: int = 224, dropout: float = 0.25):
        super().__init__()
        # Improved architecture with more features and better regularization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        flattened_dim = _infer_flatten_dim(self.conv_layers, input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))


class ResNetTransfer(nn.Module):
    """
    Transfer learning model using pretrained ResNet18.
    Much better for scene classification with limited data and many classes.
    """
    
    def __init__(self, num_classes: int, input_size: int = 224, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18 (use weights parameter for newer torchvision)
        try:
            resnet = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.DEFAULT if pretrained else None)
        except (AttributeError, TypeError):
            # Fallback for older torchvision versions
            resnet = torchvision_models.resnet18(pretrained=pretrained)
        
        # Freeze early layers (optional - can unfreeze for fine-tuning)
        # for param in list(resnet.parameters())[:-10]:
        #     param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        self.model = resnet
    
    def forward(self, x):
        return self.model(x)


class EfficientNetTransfer(nn.Module):
    """
    Transfer learning model using EfficientNet-B0.
    Even better accuracy than ResNet, but slightly slower.
    """
    
    def __init__(self, num_classes: int, input_size: int = 224, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        try:
            # Try to load EfficientNet (requires timm or torchvision >= 0.13)
            try:
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                try:
                    efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
                except (AttributeError, TypeError):
                    efficientnet = efficientnet_b0(pretrained=pretrained)
                num_features = efficientnet.classifier[1].in_features
                efficientnet.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(512, num_classes)
                )
                self.model = efficientnet
            except (ImportError, AttributeError):
                # Fallback to ResNet if EfficientNet not available
                print("EfficientNet not available, using ResNet18 instead")
                try:
                    resnet = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.DEFAULT if pretrained else None)
                except (AttributeError, TypeError):
                    resnet = torchvision_models.resnet18(pretrained=pretrained)
                num_features = resnet.fc.in_features
                resnet.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(512, num_classes)
                )
                self.model = resnet
        except Exception as e:
            raise RuntimeError(f"Could not load EfficientNet: {e}. Please install torchvision >= 0.13 or use ResNetTransfer instead.")
    
    def forward(self, x):
        return self.model(x)


__all__ = ["CNNRegularized", "ResNetTransfer", "EfficientNetTransfer"]

