import pyautogui
import time
from PIL import ImageGrab

def automate_screenshots(arrow_coordinates, screenshot_region, num_moves):
    """
    Automates taking play-by-play screenshots of games on Chess.com.
    
    :param arrow_coordinates: Tuple of (x, y) for the "Next Move" arrow button.
    :param screenshot_region: Tuple of (x1, y1, x2, y2) for the chessboard region.
    :param save_dir: Directory to save the screenshots.
    :param num_moves: Number of moves to capture.
    """
    for i in range(1, num_moves + 1):
        # Click the arrow to advance to the next move
        pyautogui.click(arrow_coordinates)
        time.sleep(0.5)  # Wait for the board to update

        # Take a screenshot of the specified area using Pillow
        try:
            screenshot = ImageGrab.grab(bbox=screenshot_region)  # bbox=(x, y, x2, y2)
            screenshot.save(f"screenshots/games/Fischer_Robatsch/screenshot_{i + 1}.png")
            print(f"Screenshot {i + 1} saved.")
        except Exception as e:
            print(f"Error capturing screenshot: {e}")

        # Short delay between moves
        time.sleep(0.5)

if __name__ == "__main__":
    # Coordinates for the "Next Move" arrow (use the mouse position tool to find these)
    arrow_coordinates = (1300, 900)  # Replace with actual coordinates
    pyautogui.click(arrow_coordinates)

    # Region of the chessboard to capture (top-left x, y, bottom-right x, y)
    screenshot_region = (240, 200, 960, 920)  # Replace with actual region coordinates
    # Take a screenshot of the specified area using Pillow
    try:
        screenshot = ImageGrab.grab(bbox=screenshot_region)  # bbox=(x, y, x2, y2)
        screenshot.save(f"screenshots/games/Fischer_Robatsch/screenshot_1.png")
        print(f"Screenshot 1 saved.")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")

    # Number of moves in the game
    num_moves = 20*2 # Replace with the actual number of moves

    automate_screenshots(arrow_coordinates, screenshot_region, num_moves)