import cv2
import os
import shutil

def split_chessboard(image_path, fen_path, output_dir, image_size=856):
    """
    Splits a chessboard image into 64 individual squares and saves them.
    Moves the corresponding FEN file to the same output folder.
    
    :param image_path: Path to the chessboard image.
    :param fen_path: Path to the corresponding FEN file.
    :param output_dir: Directory to save the individual squares.
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

    # Move the corresponding FEN file into the same folder
    shutil.copy(fen_path, os.path.join(output_dir, "board.fen"))

def process_chessboards(input_dir, fen_dir, output_dir, image_size=856):
    """
    Processes chessboard screenshots, splits them into squares, and saves the corresponding FEN files.
    
    :param input_dir: Directory containing chessboard images.
    :param fen_dir: Directory containing FEN files with matching names.
    :param output_dir: Directory to save processed boards and FEN files.
    :param image_size: The size (height/width) of the chessboard images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        if not image_name.endswith(".png"):  # Skip non-image files
            continue
        
        # Define paths
        image_path = os.path.join(input_dir, image_name)
        fen_path = os.path.join(fen_dir, os.path.splitext(image_name)[0] + ".fen")
        board_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])

        # Ensure the corresponding FEN file exists
        if not os.path.exists(fen_path):
            print(f"Warning: FEN file for {image_name} not found. Skipping...")
            continue

        # Process the chessboard
        split_chessboard(image_path, fen_path, board_output_dir, image_size)

if __name__ == "__main__":
    # Paths
    input_dir = "chessboard_screenshots"  # Directory with .png images
    fen_dir = "chessboard_screenshots"  # Directory with .fen files
    output_dir = "processed_chessboards"  # Directory to save processed boards

    # Process all chessboards
    process_chessboards(input_dir, fen_dir, output_dir)