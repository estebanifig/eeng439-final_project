import pyautogui
from PIL import ImageGrab
import pyperclip
import chess
import random
import time
import os

# Function to generate a random chess position using python-chess
def generate_random_position():
    board = chess.Board()
    board.clear()  # Start with an empty board

    # Define the required pieces
    piece_counts = {
        chess.PAWN: 6,
        chess.ROOK: 2,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.QUEEN: 1,
        chess.KING: 1
    }
    piece_colors = [chess.WHITE, chess.BLACK]

    # Place pieces for each side
    for color in piece_colors:
        available_squares = list(chess.SQUARES)  # All squares are initially available

        for piece_type, count in piece_counts.items():
            for _ in range(count):
                while True:
                    square = random.choice(available_squares)  # Pick a random square
                    if board.piece_at(square) is None:  # Ensure no overlap
                        board.set_piece_at(square, chess.Piece(piece_type, color))
                        available_squares.remove(square)  # Mark square as used
                        break

    return board.fen()

# Function to automate interaction with Chess.com and take a screenshot
def automate_chess_setup(repeat_count, fen_input_coordinates, screenshot_area):
    # Ensure the output folder exists
    os.makedirs("screenshots/random", exist_ok=True)

    # Save the initial position
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pyperclip.copy(initial_fen)

    # Input the initial FEN string
    pyautogui.click(fen_input_coordinates)
    time.sleep(0.2)
    pyautogui.hotkey('command', 'a')  # Select all
    time.sleep(0.2)
    pyautogui.hotkey('command', 'v')  # Paste the initial FEN string
    time.sleep(0.5)
    pyautogui.press('enter')  # Confirm setup
    time.sleep(2)  # Wait for the board to update

    # Take a screenshot for the initial position
    screenshot = ImageGrab.grab(bbox=screenshot_area)
    screenshot.save(f"chessboard_screenshots/initial_position.png")
    with open(f"chessboard_screenshots/initial_position.fen", "w") as fen_file:
        fen_file.write(initial_fen)
    print("Initial position saved.")

    # Save random positions
    for i in range(repeat_count):
        # Generate a random FEN string
        fen_string = generate_random_position()
        print(f"Generated FEN: {fen_string}")

        # Copy the FEN string to the clipboard
        pyperclip.copy(fen_string)

        # Input the FEN string
        pyautogui.click(fen_input_coordinates)
        time.sleep(0.2)
        pyautogui.hotkey('command', 'a')  # Select all
        time.sleep(0.2)
        pyautogui.hotkey('command', 'v')  # Paste the FEN string
        time.sleep(0.5)
        pyautogui.press('enter')  # Confirm setup
        time.sleep(2)  # Wait for the board to update

        # Take a screenshot of the specified area
        try:
            screenshot = ImageGrab.grab(bbox=screenshot_area)
            screenshot_path = f"screenshots/random/screenshot_{i + 1}.png"
            screenshot.save(screenshot_path)

            # Save the FEN string in a corresponding .fen file
            fen_path = f"chessboard_screenshots/screenshot_{i + 1}.fen"
            with open(fen_path, "w") as fen_file:
                fen_file.write(fen_string)

            print(f"Screenshot {i + 1} and FEN saved.")
        except Exception as e:
            print(f"Error capturing screenshot: {e}")

# Example usage
if __name__ == "__main__":
    # Screen coordinates for the FEN input box (use the first script to find these)
    fen_input_coordinates = (-700, -615)  # Replace with actual coordinates of FEN input box

    # Area for the screenshot (x1, y1, x2, y2)
    screenshot_area = (-1822, -918, -966, -62)  # Replace with valid coordinates for the chessboard

    # Run the automation for 99 random positions + 1 initial position
    automate_chess_setup(repeat_count=499, fen_input_coordinates=fen_input_coordinates, screenshot_area=screenshot_area)