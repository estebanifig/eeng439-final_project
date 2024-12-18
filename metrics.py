import torch
import json
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from src.dataset_loader import load_datasets  # Assuming you already have this
from src.model import create_efficientnet  # Assuming you already have this
import os
import random
import chess.svg
import chess
import cairosvg

def load_model(model_path, class_indices_path, device):
    """
    Load the trained EfficientNet model and class indices.
    """
    with open(class_indices_path, "r") as f:
        class_names = json.load(f)

    model = create_efficientnet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image, device):
    """
    Preprocess a single image for model inference.
    """
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def evaluate_square_level_accuracy(model, dataloader, device):
    """
    Evaluate square-level accuracy on the test dataset.
    """
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1


def evaluate_fen_reconstruction(model, class_names, test_fens, device):
    """
    Evaluate FEN reconstruction accuracy using test FEN strings.
    """
    correct = 0

    for fen_data in test_fens:
        image_path, true_fen = fen_data
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess_image(image, device)

        squares = []
        square_size = tensor.size(-1) // 8

        for row in range(8):
            row_squares = []
            for col in range(8):
                square_tensor = tensor[:, :, row*square_size:(row+1)*square_size, col*square_size:(col+1)*square_size]
                output = model(square_tensor)
                _, predicted = torch.max(output, 1)
                row_squares.append(class_names[predicted.item()])
            squares.append(row_squares)

        predicted_fen = generate_fen(squares)
        if predicted_fen == true_fen:
            correct += 1

    fen_accuracy = correct / len(test_fens)
    return fen_accuracy


def generate_fen(squares):
    """
    Convert square predictions to a FEN string.
    """
    piece_to_fen = {
        "empty": "1",
        "white_pawn": "P",
        "black_pawn": "p",
        "white_rook": "R",
        "black_rook": "r",
        "white_knight": "N",
        "black_knight": "n",
        "white_bishop": "B",
        "black_bishop": "b",
        "white_queen": "Q",
        "black_queen": "q",
        "white_king": "K",
        "black_king": "k",
    }
    fen_rows = []
    for row in squares:
        fen_row = ""
        empty_count = 0
        for piece in row:
            if piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_to_fen[piece]
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w - - 0 1"


def generate_test_fens(output_path, num_entries=50):
    """
    Generate a JSON file with test FEN strings and corresponding synthetic images.

    Parameters:
        output_path (str): Path to save the JSON file.
        num_entries (int): Number of FEN entries to generate.
    """
    fen_data = []
    os.makedirs("synthetic_images", exist_ok=True)  # Ensure directory exists

    for _ in range(num_entries):
        board = chess.Board()
        board.clear()  # Clear the board
        for _ in range(random.randint(5, 20)):  # Add random pieces
            square = random.choice(list(chess.SQUARES))
            piece = random.choice(list(chess.PIECE_SYMBOLS[1:]))  # Exclude king for simplicity
            color = random.choice([chess.WHITE, chess.BLACK])
            board.set_piece_at(square, chess.Piece.from_symbol(piece.upper() if color else piece))

        fen = board.fen()
        image_path = f"synthetic_images/{random.randint(1, 100000)}.png"
        
        # Convert SVG to PNG
        svg_data = chess.svg.board(board=board)
        cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to=image_path)
        
        fen_data.append({"image_path": image_path, "true_fen": fen})

    with open(output_path, "w") as f:
        json.dump(fen_data, f, indent=4)
    print(f"Generated {num_entries} FEN entries in {output_path}")


def main():
    """
    Main function to run all metrics evaluation.
    """
    # Paths
    model_path = "models/final_chess_efficientnet.pth"
    class_indices_path = "models/class_indices.json"
    test_data_dir = "data/split_dataset/test"  # Directory for test squares
    test_fens_file = "test_fens.json"  # JSON file containing image paths and true FEN strings

    # Generate test FENs if file doesn't exist
    if not os.path.exists(test_fens_file):
        print("Generating FEN JSON file...")
        generate_test_fens(test_fens_file)

    # Check for device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and class names
    model, class_names = load_model(model_path, class_indices_path, device)

    # Load test data
    _, _, test_loader, _ = load_datasets(None, None, test_data_dir, batch_size=32)

    # Evaluate square-level accuracy
    square_accuracy, f1_score_result = evaluate_square_level_accuracy(model, test_loader, device)
    print(f"Square-Level Accuracy: {square_accuracy * 100:.2f}%")
    print(f"F1 Score: {f1_score_result:.2f}")

    # Evaluate FEN reconstruction accuracy
    # with open(test_fens_file, "r") as f:
    #     test_fens = json.load(f)
    # fen_accuracy = evaluate_fen_reconstruction(model, class_names, test_fens, device)
    # print(f"FEN Reconstruction Accuracy: {fen_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()