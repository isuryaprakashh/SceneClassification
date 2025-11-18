import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models as pt_models

class BinaryClassifier(nn.Module):

    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class MultiClassClassifier(nn.Module):

    def __init__(self, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32x32 -> 16x16
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

class CNNClassifier_regularization(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier_regularization, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32 -> 16
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),   # dropout for regularization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        # Load pretrained VGG16
        self.model = pt_models.vgg16(pretrained=True)
        # Freeze feature extractor layers (optional)
        for param in self.model.features.parameters():
            param.requires_grad = False
            # Replace final classifier layer dynamically
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetClassifier, self).__init__()
        # Load pretrained AlexNet
        self.model = pt_models.alexnet(pretrained=True)
        # Freeze feature extractor layers (optional)
        for param in self.model.features.parameters():
            param.requires_grad = False
            # Replace final classifier layer dynamically
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, encoded_dim=256):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 56 -> 28
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*28*28, encoded_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 256*28*28),
            nn.ReLU(),
            nn.Unflatten(1, (256, 28, 28)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0), # 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), # 224
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, encoded_dim=256):
        super(ResNetClassifier, self).__init__()
        # Use ResNet18 as feature extractor
        self.resnet = pt_models.resnet18(weights=None)
        # Modify first layer to take encoded feature input
        self.resnet.fc = nn.Linear(512, num_classes)

        # Optional adapter if encoded features are directly used
        self.feature_adapter = nn.Linear(encoded_dim, 512)

    def forward(self, encoded_features):
        # Convert latent vector into ResNet-compatible feature space
        adapted_features = self.feature_adapter(encoded_features)
        # Classification
        out = self.resnet.fc(adapted_features)
        return out

class EncoderClassifierWrapper(nn.Module):
    def __init__(self, autoencoder, classifier):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.classifier = classifier

    def forward(self, x):
        with torch.no_grad():  # freeze encoder if desired
            encoded = self.encoder(x)
        out = self.classifier(encoded)
        return out

def get_model(task_type, num_classes=None):
    """Returns the model based on classification type."""
    if task_type == "binary":
        return BinaryClassifier()
    elif task_type == "multi":
        if num_classes is None:
            raise ValueError("num_classes must be specified for multi-class classification.")
        return MultiClassClassifier(num_classes)
    else:
        raise ValueError("Invalid task type. Choose 'binary' or 'multi'.")

def get_optimizer(optimizer_name, model, lr=0.001):
    """Returns an optimizer based on the provided name."""
    if optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name. Choose 'SGD', 'Adam', or 'RMSprop'.")

def visualize_layer_outputs(model, input_image, layers_to_visualize=None, max_features=16):

    model.eval()
    x = input_image.to(next(model.parameters()).device)

    if layers_to_visualize is None:
        layers_to_visualize = [0, 3, 6]

    with torch.no_grad():
        for idx, layer in enumerate(model.conv_layers):
            x = layer(x)
            if idx in layers_to_visualize:
                num_features = min(x.shape[1], max_features)
                plt.figure(figsize=(12, 6))
                for i in range(num_features):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(x[0, i].cpu(), cmap='viridis')
                    plt.axis('off')
                plt.suptitle(f'Layer {idx} Feature Maps')
                plt.show()

def visualize_reconstruction(model, dataloader, device, num_images=6):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)

    with torch.no_grad():
        encoded, outputs = model(images)   # <-- Unpack tuple here

    # Move to CPU
    images = images.cpu()
    outputs = outputs.cpu()

    # De-normalize from [-1, 1] → [0, 1]
    images = (images + 1) / 2
    outputs = (outputs + 1) / 2

    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i].permute(1, 2, 0).squeeze())
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(outputs[i].permute(1, 2, 0).squeeze())
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.show()

