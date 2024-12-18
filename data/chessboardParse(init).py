import cv2
import os

def split_chessboard(image_path, output_dir, initial_fen, image_size=856):
    """
    Splits a chessboard image into 64 individual squares and saves them.
    Saves the hardcoded FEN file into the same output folder.
    
    :param image_path: Path to the chessboard image.
    :param output_dir: Directory to save the individual squares.
    :param initial_fen: The standard initial FEN string.
    :param image_size: The size (height/width) of the chessboard image.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    square_size = image_size // 8  # Assuming square chessboard

    # Split the board row by row and column by column
    for row in range(8):
        for col in range(8):
            # Extract each square
            y_start = row * square_size
            y_end = (row + 1) * square_size
            x_start = col * square_size
            x_end = (col + 1) * square_size
            square = image[y_start:y_end, x_start:x_end]

            # Convert row/col to chess notation
            rank = 8 - row  # Convert row index to rank (8, 7, ..., 1)
            file = chr(ord('a') + col)  # Convert column index to file (a, b, ..., h)
            square_name = f"{file}{rank}.png"  # e.g., a8.png, b7.png

            # Save the square
            cv2.imwrite(os.path.join(output_dir, square_name), square)

    # Save the hardcoded initial FEN string into the folder
    fen_output_path = os.path.join(output_dir, "board.fen")
    with open(fen_output_path, "w") as fen_file:
        fen_file.write(initial_fen)

def process_chessboards(input_dir, output_dir, image_size=856):
    """
    Processes chessboard screenshots, splits them into squares, and saves the hardcoded FEN file.
    
    :param input_dir: Directory containing chessboard images.
    :param output_dir: Directory to save processed boards and FEN files.
    :param image_size: The size (height/width) of the chessboard images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Standard initial FEN string
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    for image_name in os.listdir(input_dir):
        if not image_name.endswith(".png"):  # Skip non-image files
            continue
        
        # Define paths
        image_path = os.path.join(input_dir, image_name)
        board_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])

        # Process the chessboard
        split_chessboard(image_path, board_output_dir, initial_fen, image_size)

if __name__ == "__main__":
    # Paths
    input_dir = "data/chessboard_screenshots"  # Directory with .png images
    output_dir = "data/processed_chessboards"  # Directory to save processed boards

    # Process all chessboards
    process_chessboards(input_dir, output_dir)