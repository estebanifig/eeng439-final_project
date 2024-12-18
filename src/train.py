import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
import os
import json

from .dataset_loader import load_datasets
from .model import create_efficientnet

def compute_class_weights(train_dataset):
    """
    Computes class weights to handle class imbalance.

    Parameters:
        train_dataset (ImageFolder): Training dataset object.

    Returns:
        torch.Tensor: Class weights as a tensor.
    """
    class_counts = Counter([label for _, label in train_dataset.samples])
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    # Compute inverse frequency for each class
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(train_dir, val_dir, save_path, epochs=10, batch_size=32, lr=0.001):
    """
    Trains the EfficientNet model with a given dataset and saves the best model.
    Implements class weights to handle class imbalance.

    Parameters:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        save_path (str): Path to save the best model's weights.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for the DataLoader.
        lr (float): Learning rate for the optimizer.

    Returns:
        None
    """
    # Check for MPS (Apple Silicon GPU) availability; fall back to CPU if not found
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        print("MPS device not found. Using CPU.")
        device = torch.device("cpu")
    
    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load data loaders
    train_loader, val_loader, _, classes = load_datasets(train_dir, val_dir, None, batch_size)
    print(f"Classes: {classes}")

    # Save class indices (order)
    with open("models/class_indices.json", "w") as f:
        json.dump(classes, f)
        print("Saved class indices to models/class_indices.json")

    # Compute class weights
    train_dataset, _, _, _ = load_datasets(train_dir, None, None, batch_size)
    class_weights = compute_class_weights(train_dataset.dataset).to(device)
    print(f"Class Weights: {class_weights}")

    # Load pre-trained EfficientNet and modify for the number of classes
    model = create_efficientnet(num_classes=len(classes)).to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track best validation accuracy
    best_val_acc = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

if __name__ == "__main__":
    train_model("split_dataset/train", "split_dataset/val", "models/final_chess_efficientnet.pth")