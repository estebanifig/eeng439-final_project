import sys
import os

sys.path.append(os.path.abspath("src"))

from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    # Paths
    train_dir = "data/split_dataset/train"
    val_dir = "data/split_dataset/val"
    test_dir = "data/split_dataset/test"
    model_path = "models/final_chess_efficientnet.pth"

    # Train the model
    print("Starting training...")
    train_model("data/split_dataset/train", "data/split_dataset/val", "models/final_chess_efficientnet.pth")

    # Evaluate the model
    print("Starting evaluation...")
    evaluate_model(test_dir, model_path)