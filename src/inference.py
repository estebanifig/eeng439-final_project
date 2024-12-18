import torch
from torchvision import transforms
from PIL import Image
import json
from model import create_efficientnet

def predict_square(image_path, model_path, class_indices_path):
    """
    Predict the class of a single chess square image.

    Parameters:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model file.
        class_indices_path (str): Path to the saved class indices JSON.
    """
    # Image transformation pipeline
    transform_inference = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(           # Normalize to [-1, 1] range for each channel
            mean=[0.5, 0.5, 0.5],       # Mean pixel values for R, G, B
            std=[0.5, 0.5, 0.5]         # Standard deviation for R, G, B
        )
    ])
    
    # Load the class indices
    with open(class_indices_path, "r") as f:
        class_names = json.load(f)
    
    # Load the model
    model = create_efficientnet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform_inference(image).unsqueeze(0)
    
    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]

if __name__ == "__main__":
    """
    Example usage: Predict the class of a single chess square image.
    """
    # Paths to the image, model, and saved class indices
    image_path = "data/database/white_rook/a1_001.png"
    model_path = "models/final_chess_efficientnet.pth"
    class_indices_path = "models/class_indices.json"

    # Run prediction and print the result
    result = predict_square(image_path, model_path, class_indices_path)
    print("Predicted class:", result)