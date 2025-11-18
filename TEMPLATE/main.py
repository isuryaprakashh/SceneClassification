import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import torch
import torch.nn as nn
import torch.optim as optim
import Preprocess as pre
import models1 as models
import random
import resultsplot as rp
def train(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {accuracy:.2f}%")

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss=val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.2f}, Validation Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss

def evaluate_random_minibatch(model, val_loader, criterion, device, num_batches=1):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    # Pick random batch indices
    all_batches = list(val_loader)
    chosen_batches = random.sample(all_batches, min(num_batches, len(all_batches)))

    with torch.no_grad():
        for images, labels in chosen_batches:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_accuracy = 100 * correct / total
    avg_loss = val_loss / len(chosen_batches)
    print(f"Random Mini-Batch Eval → Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
    return avg_accuracy, avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_to_data = "dataset"
    train_loader, val_loader, classes = pre.load_data(data_dir=path_to_data, batch_size=32, augment=True)

    print(f"\nNumber of classes in dataset: {len(classes)}")
    print(f"Class names: {classes}")
    print(f"Size of training set: {len(train_loader.dataset)}")
    print(f"Size of validation set: {len(val_loader.dataset)}")

    print("\nDisplaying Sample Images from Dataset...")
    pre.visualize_data(train_loader, classes)

    task_type = "binary" if len(classes) == 2 else "multi"
    num_classes = len(classes) if task_type == "multi" else None
    """model = models.get_model(task_type, num_classes=num_classes).to(device)

    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining the model...")
    #train(model, train_loader, optimizer, criterion, device, epochs=10)

    #print("\nEvaluating on validation set...")
    #evaluate(model, val_loader, criterion, device)

    results = {}

    for opt_name in ["SGD", "Adam", "RMSprop"]:
        print(f"\n===== Training with {opt_name} Optimizer =====")
        model = models.get_model(task_type, num_classes=num_classes).to(device)
        optimizer = models.get_optimizer(opt_name, model, lr=0.001)

        print("\nTraining the model...")
        train(model, train_loader, optimizer, criterion, device, epochs=10)

        print("\nEvaluating on validation set...")
        acc,loss = evaluate(model, val_loader, criterion, device)

        results[opt_name] = {"accuracy": acc, "loss": loss}
    #rp.plot_optimizer_results(results)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining the model...")
    train(model, train_loader, optimizer, criterion, device, epochs=10)

    print("\nEvaluating on random minibatch validation set...")
    evaluate_random_minibatch(model, val_loader, criterion, device)

    model = models.CNNClassifier(num_classes=num_classes).to(device)
    print(model)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model, val_loader, criterion, device)
    evaluate_random_minibatch(model, val_loader, criterion, device)

    # Visualize layer-wise outputs
    print("\nVisualizing CNN layer outputs for a sample image...")
    sample_image, _ = next(iter(val_loader))
    sample_image = sample_image[0].unsqueeze(0)
    models.visualize_layer_outputs(model, sample_image)

    model1 = models.CNNClassifier(num_classes=num_classes).to(device)
    print(model1)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    print("CNN classifier training")
    train(model1, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model1, val_loader, criterion, device)

    model2 = models.CNNClassifier_regularization(num_classes=num_classes).to(device)
    print(model2)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
    print("CNN classifier with regularization training")
    train(model2, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model2, val_loader, criterion, device)

    # VGG16 model
    vgg_model = models.VGGClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)
    print("VGG16 model training...")
    train(vgg_model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(vgg_model, val_loader, criterion, device)

    # AlexNet model
    alex_model = models.AlexNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(alex_model.parameters(), lr=0.001)
    print("AlexNet Model Training...")
    train(alex_model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(alex_model, val_loader, criterion, device)"""
    # --- Encoder + Classifier Wrapper ---
    autoencoder = models.AutoEncoder(encoded_dim=256)
    autoencoder.eval()  # freeze encoder
    classifier = models.ResNetClassifier(num_classes, encoded_dim=256)
    model = models.EncoderClassifierWrapper(autoencoder, classifier).to(device)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training Encoder-Classifier Wrapper...")
    train(model, train_loader, optimizer, criterion, device, epochs=10)
    evaluate(model, val_loader, criterion, device)

    print("\nVisualizing original and reconstructed images from AutoEncoder...")
    models.visualize_reconstruction(autoencoder, val_loader, device, num_images=6)

if __name__ == "__main__":
    main()