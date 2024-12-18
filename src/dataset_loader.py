import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_datasets(train_dir=None, val_dir=None, test_dir=None, batch_size=32):
    """
    Loads datasets for training, validation, and testing.
    Handles cases where train, val, or test directories may be None.

    Parameters:
        train_dir (str): Path to the training dataset directory.
        val_dir (str): Path to the validation dataset directory.
        test_dir (str): Path to the test dataset directory (optional).
        batch_size (int): Number of samples per batch for the DataLoader.

    Returns:
        train_loader (DataLoader or None): DataLoader for the training dataset.
        val_loader (DataLoader or None): DataLoader for the validation dataset.
        test_loader (DataLoader or None): DataLoader for the test dataset, if provided.
        classes (list): List of class names, if available; otherwise, an empty list.
    """
    # Augmentation and preprocessing for training data
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),      # Randomly rotate image by ±10 degrees
        transforms.RandomHorizontalFlip(),  # Randomly horizontally flip the image
        transforms.Resize((224, 224)),      # Resize to 224x224
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Preprocessing for validation and test data (no augmentations)
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),      # Resize to 224x224
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Initialize loaders and class names
    train_loader = val_loader = test_loader = None
    class_names = []

    # Load training dataset if provided
    if train_dir:
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        class_names = train_dataset.classes  # Get class names

    # Load validation dataset if provided
    if val_dir:
        val_dataset = datasets.ImageFolder(val_dir, transform=transform_val_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        class_names = val_dataset.classes  # Overwrite only if train_dir not provided

    # Load test dataset if provided
    if test_dir:
        test_dataset = datasets.ImageFolder(test_dir, transform=transform_val_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        class_names = test_dataset.classes

    return train_loader, val_loader, test_loader, class_names