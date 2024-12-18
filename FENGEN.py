import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
from PIL import Image
import chess
import chess.engine
import json
import time
import mss
import mss.tools

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS if available, otherwise default to CPU
print(f"Using device: {DEVICE}")

# Load the model
def load_model(model_path, class_indices_path):
    """
    Loads the EfficientNet model and the corresponding class indices.

    Parameters:
        model_path (str): Path to the trained model weights.
        class_indices_path (str): Path to the JSON file containing class indices.

    Returns:
        model (torch.nn.Module): The loaded EfficientNet model.
        class_names (list): List of class names corresponding to the model's output.
    """
    # Load class names
    with open(class_indices_path, "r") as f:
        class_names = json.load(f)
    
    # Load EfficientNet model with custom classifier for chess pieces
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode
    return model, class_names

# Preprocess chessboard image
def preprocess_square(image):
    """
    Preprocesses an individual square image for model prediction.

    Parameters:
        image (PIL.Image.Image): Image of the chessboard square.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 (model input size)
        transforms.ToTensor(),         # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
    ])
    return transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension and send to device

# Validate FEN string
def is_valid_fen(fen):
    """
    Validates the given FEN string and checks if the board is not empty.

    Parameters:
        fen (str): The FEN string representing the chessboard.

    Returns:
        bool: True if the FEN is valid and the board is not empty, False otherwise.
    """
    try:
        board = chess.Board(fen)
        # Check if all squares are empty
        is_empty = all(board.piece_at(square) is None for square in chess.SQUARES)
        return not is_empty  # Return False if the board is empty
    except ValueError:
        return False  # Invalid FEN

# Detect chessboard and generate FEN
def detect_chessboard(frame, model, class_names):
    """
    Detects a chessboard in the frame and generates its FEN representation.

    Parameters:
        frame (numpy.ndarray): The input video frame.
        model (torch.nn.Module): The trained EfficientNet model.
        class_names (list): List of class names for the model.

    Returns:
        str or None: The FEN string if a valid chessboard is detected, None otherwise.
    """
    # Convert frame to grayscale and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Check for quadrilateral (chessboard-like shape)
            x, y, w, h = cv2.boundingRect(approx)
            roi = frame[y:y+h, x:x+w]

            if w > 200 and h > 200:  # Threshold to filter out small detections
                roi = cv2.resize(roi, (856, 856))  # Resize to match expected chessboard size
                squares = []

                # Split the chessboard into individual squares
                square_size = 856 // 8
                for row in range(8):
                    row_squares = []
                    for col in range(8):
                        y_start, y_end = row * square_size, (row + 1) * square_size
                        x_start, x_end = col * square_size, (col + 1) * square_size
                        square = roi[y_start:y_end, x_start:x_end]
                        square_pil = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
                        tensor = preprocess_square(square_pil)

                        # Predict the class for the square
                        with torch.no_grad():
                            output = model(tensor)
                            _, predicted = torch.max(output, 1)
                            row_squares.append(class_names[predicted.item()])
                    squares.append(row_squares)

                # Convert the detected squares to FEN
                return generate_fen(squares)
    return None

# Generate FEN string
def generate_fen(squares):
    """
    Generates a FEN string from the model's predictions of chessboard squares.

    Parameters:
        squares (list): 8x8 list of predicted classes for each square.

    Returns:
        str: The FEN string representing the chessboard position.
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
    return "/".join(fen_rows) + " w - - 0 1"  # Append default turn and castling rights

# Get optimal move using Stockfish
def get_optimal_move(fen):
    """
    Gets the optimal move from the current FEN using Stockfish.

    Parameters:
        fen (str): The FEN string representing the chessboard.

    Returns:
        str: The optimal move in algebraic notation or an error message.
    """
    stockfish_path = "src/stockfish/stockfish-macos-m1-apple-silicon"  # Path to Stockfish executable

    # Validate the FEN string
    if not is_valid_fen(fen):
        return "Invalid FEN or empty board"

    # Use Stockfish to get the optimal move
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(time=1.0))  # Set time limit for move calculation
        return result.move

# Main function for real-time chessboard detection
def main():
    """
    Captures the screen in real-time, detects chessboards, generates FEN strings, and computes optimal moves.
    """
    model_path = "models/final_chess_efficientnet.pth"  # Path to the trained model weights
    class_indices_path = "models/class_indices.json"   # Path to class indices JSON

    # Load model and class names
    model, class_names = load_model(model_path, class_indices_path)

    # Set up screen capture
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary monitor
        print("Starting real-time screen capture...")

        last_fen = None  # Store the last detected FEN string

        while True:
            # Capture the screen
            screen = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            # Detect chessboard and generate FEN
            fen = detect_chessboard(frame, model, class_names)
            if fen and fen != last_fen and is_valid_fen(fen):  # Check for valid, updated FEN
                print(f"Detected FEN: {fen}")
                last_fen = fen

                # Get and print the optimal move
                optimal_move = get_optimal_move(fen)
                print(f"Optimal move: {optimal_move}")

            # Display the video frame
            cv2.imshow("Chessboard Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()