import torch
from tqdm import tqdm  # Import tqdm for progress bar
from dataset_loader import load_datasets  # Function to load datasets
from model import create_efficientnet     # Function to create the EfficientNet model

def evaluate_model(test_dir, model_path):
    """
    Evaluates a trained model on the test dataset and calculates accuracy.

    Parameters:
        test_dir (str): Path to the test dataset directory.
        model_path (str): Path to the trained model file (.pth).
    """

    # Check for MPS (Apple Silicon GPU) availability; fall back to CPU if not found
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS backend
        print(f"Using device: {device}")
    else:
        print("MPS device not found. Using CPU.")
        device = torch.device("cpu")  # Fall back to CPU if MPS is unavailable

    # Load the test dataset using the load_datasets function
    _, _, test_loader, classes = load_datasets(None, None, test_dir)

    # Initialize the model architecture and load the saved weights
    model = create_efficientnet(num_classes=len(classes))  # Create model with correct output size
    model.load_state_dict(torch.load(model_path))          # Load trained model weights
    model = model.to(device)                              # Move the model to the selected device (MPS or CPU)

    # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()

    # Initialize counters for accuracy calculation
    correct = 0  # Total correct predictions
    total = 0    # Total number of samples

    # Use tqdm to show a progress bar during evaluation
    print("\nStarting evaluation...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating", leave=True)
        for inputs, labels in test_bar:
            # Move the inputs and labels to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Perform forward pass: compute predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs, 1)  # Returns (max values, indices)

            # Update total sample count and correct prediction count
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar with the current accuracy
            test_bar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    # Final accuracy
    accuracy = 100 * correct / total
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    """
    Entry point for the script. 
    Specifies the test dataset directory and model file to use for evaluation.
    """
    evaluate_model(
        test_dir="split_dataset/test",                # Path to the test dataset
        model_path="models/final_chess_efficientnet.pth"  # Path to the trained model weights
    )