import torch.nn as nn
from torchvision import models

def create_efficientnet(num_classes=13):
    """
    Creates an EfficientNet-B0 model pre-trained on ImageNet and modifies the final layer 
    for a custom number of output classes.

    Parameters:
        num_classes (int): The number of output classes for classification.
                          Default is 13 (e.g., chess piece types + empty squares).

    Returns:
        model (torchvision.models.EfficientNet): The modified EfficientNet-B0 model.
    """
    # Load pre-trained EfficientNet-B0 weights
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1  # Pre-trained on ImageNet
    model = models.efficientnet_b0(weights=weights)         # Load EfficientNet-B0 model with weights

    # Modify the classifier layer
    # The EfficientNet-B0 model has a classifier attribute structured as:
    # nn.Sequential(
    #     nn.Dropout(p=0.2, inplace=True),
    #     nn.Linear(in_features, 1000)  # Pre-trained output for 1000 ImageNet classes
    # )

    # Get the number of input features to the final fully connected layer
    num_ftrs = model.classifier[1].in_features

    # Replace the final linear layer to match the desired number of classes
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Return the modified model
    return model