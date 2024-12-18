import os
import shutil
import chess
from collections import defaultdict

def parse_fen(fen_path):
    """
    Parses the FEN file and returns a mapping of squares to piece classes.
    
    :param fen_path: Path to the FEN file.
    :return: Dictionary mapping square positions (e.g., 'a1') to classes (e.g., 'white_rook', 'empty').
    """
    with open(fen_path, 'r') as fen_file:
        fen_string = fen_file.read().strip()
    
    board = chess.Board(fen_string)
    square_classes = {}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        square_name = chess.square_name(square)  # e.g., 'a1', 'b2'
        if piece:
            color = 'white' if piece.color else 'black'
            piece_type = chess.PIECE_NAMES[piece.piece_type]
            square_classes[square_name] = f"{color}_{piece_type}"  # e.g., 'white_pawn'
        else:
            square_classes[square_name] = 'empty'

    return square_classes

def sort_squares(processed_chessboard_dir, database_dir):
    """
    Sorts all squares from processed_chessboard into class folders inside database.
    
    :param processed_chessboard_dir: Path to the processed chessboard directory.
    :param database_dir: Path to the database directory with class folders.
    """
    # Ensure all class folders exist in the database
    class_folders = [
        "empty", "white_pawn", "black_pawn", "white_rook", "black_rook",
        "white_knight", "black_knight", "white_bishop", "black_bishop",
        "white_queen", "black_queen", "white_king", "black_king"
    ]
    for class_name in class_folders:
        os.makedirs(os.path.join(database_dir, class_name), exist_ok=True)
    
    # Track unique naming for each class and position
    name_counters = defaultdict(int)

    # Iterate through all folders in processed_chessboard
    for board_folder in os.listdir(processed_chessboard_dir):
        board_path = os.path.join(processed_chessboard_dir, board_folder)
        if not os.path.isdir(board_path):
            continue

        # Find the corresponding FEN file
        fen_path = os.path.join(board_path, "board.fen")
        if not os.path.exists(fen_path):
            print(f"Warning: FEN file missing for {board_folder}. Skipping...")
            continue

        # Parse the FEN file to get square classes
        square_classes = parse_fen(fen_path)

        # Process each square in the board folder
        for square_file in os.listdir(board_path):
            if not square_file.endswith(".png"):  # Skip non-image files
                continue

            # Extract the square position (e.g., 'a1' from 'a1.png')
            square_name = os.path.splitext(square_file)[0]

            # Get the class for the square
            if square_name not in square_classes:
                print(f"Warning: {square_name} not found in FEN for {board_folder}. Skipping...")
                continue
            square_class = square_classes[square_name]

            # Determine the destination folder and unique name
            dest_folder = os.path.join(database_dir, square_class)
            name_counters[square_name] += 1
            dest_file_name = f"{square_name}_{name_counters[square_name]:03d}.png"
            dest_path = os.path.join(dest_folder, dest_file_name)

            # Move the square to the correct class folder
            src_path = os.path.join(board_path, square_file)
            shutil.move(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")

if __name__ == "__main__":
    # Paths
    processed_chessboard_dir = "processed_chessboards"  # Folder containing all processed chessboards
    database_dir = "database"  # Folder containing 13 class folders

    # Sort squares into the database
    sort_squares(processed_chessboard_dir, database_dir)