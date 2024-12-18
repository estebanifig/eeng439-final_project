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
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load the model
def load_model(model_path, class_indices_path):
    """
    Loads the EfficientNet model and the corresponding class indices.
    """
    with open(class_indices_path, "r") as f:
        class_names = json.load(f)
    
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, class_names

# Preprocess chessboard image
def preprocess_square(image):
    """
    Preprocesses an individual square image for model prediction.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Validate FEN string
def is_valid_fen(fen):
    """
    Validates the given FEN string and checks if the position is realistic.
    """
    try:
        board = chess.Board(fen)
        return board.is_valid()
    except ValueError:
        return False

# Detect chessboard and generate FEN
def detect_chessboard(frame, model, class_names):
    """
    Detects a chessboard in the frame and generates its FEN representation.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            roi = frame[y:y+h, x:x+w]

            if w > 200 and h > 200:
                roi = cv2.resize(roi, (856, 856))
                squares = []
                square_size = 856 // 8
                for row in range(8):
                    row_squares = []
                    for col in range(8):
                        y_start, y_end = row * square_size, (row + 1) * square_size
                        x_start, x_end = col * square_size, (col + 1) * square_size
                        square = roi[y_start:y_end, x_start:x_end]
                        square_pil = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
                        tensor = preprocess_square(square_pil)

                        with torch.no_grad():
                            output = model(tensor)
                            _, predicted = torch.max(output, 1)
                            row_squares.append(class_names[predicted.item()])
                    squares.append(row_squares)

                return generate_fen(squares)
    return None

# Generate FEN string
def generate_fen(squares):
    """
    Generates a FEN string from the model's predictions of chessboard squares.
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
                fen_row += piece_to_fen.get(piece, "?")
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w - - 0 1"

# Get optimal move using Stockfish
def get_optimal_move(fen):
    """
    Gets the optimal move from the current FEN using Stockfish.
    """
    stockfish_path = "src/stockfish/stockfish-macos-m1-apple-silicon"

    if not is_valid_fen(fen):
        return "Invalid FEN or empty board"

    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            result = engine.play(board, chess.engine.Limit(time=1.0))
            return result.move
    except chess.engine.EngineTerminatedError:
        return "Stockfish terminated unexpectedly"
    except Exception as e:
        return f"Error: {str(e)}"

# Main function
def main():
    model_path = "models/final_chess_efficientnet.pth"
    class_indices_path = "models/class_indices.json"
    model, class_names = load_model(model_path, class_indices_path)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        print("Starting real-time screen capture...")

        last_fen = None

        while True:
            screen = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            fen = detect_chessboard(frame, model, class_names)
            if fen and fen != last_fen and is_valid_fen(fen):
                print(f"Detected FEN: {fen}")
                last_fen = fen
                optimal_move = get_optimal_move(fen)
                print(f"Optimal move: {optimal_move}")
            elif fen and not is_valid_fen(fen):
                print(f"Invalid FEN detected: {fen}, skipping...")

            cv2.imshow("Chessboard Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()