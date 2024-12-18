import pyautogui
from PIL import ImageGrab
import time
import os

# Function to automate interaction with Chess.com and take screenshots
def automate_chess_setup(start_index, repeat_count, screenshot_area):
    """
    Takes screenshots of the board and pieces without altering the FEN position.
    """
    for i in range(repeat_count):  # Take `repeat_count` screenshots

        # Take a screenshot
        screenshot_path = f"chessboard_screenshots/screenshot_{start_index + i}.png"
        try:
            screenshot = ImageGrab.grab(bbox=screenshot_area)
            screenshot.save(screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"Error capturing screenshot: {e}")

# Function to change the board or piece theme
def change_theme(settings_coordinates, tab_coordinates, dropdown_coordinates, exit_coordinates):
    """
    Changes the board or piece theme by navigating through the settings menu.
    """
    # Click the settings button
    pyautogui.moveTo(settings_coordinates)
    time.sleep(0.1)
    pyautogui.click(settings_coordinates)
    time.sleep(0.1)  # Wait for the settings menu to open

    # Click the boards tab
    pyautogui.moveTo(tab_coordinates)
    time.sleep(0.1)
    pyautogui.click(tab_coordinates)
    time.sleep(0.1)  # Wait for the boards tab to load

    # Open the dropdown menu
    pyautogui.moveTo(dropdown_coordinates)
    time.sleep(0.1)
    pyautogui.click(dropdown_coordinates)
    time.sleep(0.1)

    # Navigate the dropdown menu
    pyautogui.press("down")
    time.sleep(0.2)

    pyautogui.press("enter")
    time.sleep(0.2)

    # Exit the settings menu
    pyautogui.moveTo(exit_coordinates)
    time.sleep(0.1)
    pyautogui.click(exit_coordinates)
    time.sleep(0.5)  # Allow time for the board to update

# Function to reset the dropdown menu to the first option
def reset_to_first_option(up_arrow_count):
    """
    Sends the up arrow key `up_arrow_count` times to return to the first option in the dropdown menu.
    """
    for _ in range(up_arrow_count):
        pyautogui.press("up")
        time.sleep(0.1)  # Short delay between key presses

if __name__ == "__main__":
    # Configuration
    screenshot_area = (-1822, -918, -966, -62)  # Replace with valid coordinates for the chessboard
    settings_coordinates = (-950, -950)  # Coordinates to click the settings button
    tab_coordinates = (-1050, -790)  # Coordinates to click the boards tab
    board_dropdown_coordinates = (-1130, -690)  # Coordinates to open the board dropdown menu
    piece_dropdown_coordinates = (-1130, -741)  # Coordinates to open the piece dropdown menu
    exit_coordinates = (-965, -800)  # Coordinates to exit the settings menu
    origin = (-700, -400)
    pyautogui.moveTo(origin)
    time.sleep(0.2)
    pyautogui.click(origin)
    time.sleep(0.2)

    # Number of screenshots per board/piece theme
    repeat_count = 10  # Number of screenshots per theme

    # Total number of screenshots
    screenshot_index = 1

    for piece_theme in range(36):  # 36 chess piece themes
        for board_theme in range(31):  # 31 board themes

            # Capture screenshots for this board + piece combination
            automate_chess_setup(
                start_index=screenshot_index,
                repeat_count=repeat_count,
                screenshot_area=screenshot_area,
            )

            change_theme(
                settings_coordinates,
                tab_coordinates,
                board_dropdown_coordinates,  # Use board dropdown coordinates
                exit_coordinates,
            )

            # Update the screenshot index dynamically
            screenshot_index += repeat_count

        # Reset board themes to the first option after completing all board themes
        # Click the settings button
        pyautogui.moveTo(settings_coordinates)
        time.sleep(0.1)
        pyautogui.click(settings_coordinates)
        time.sleep(0.1)  # Wait for the settings menu to open

        # Click the boards tab
        pyautogui.moveTo(tab_coordinates)
        time.sleep(0.1)
        pyautogui.click(tab_coordinates)
        time.sleep(0.1)  # Wait for the boards tab to load

        # Open the dropdown menu
        pyautogui.moveTo(board_dropdown_coordinates)
        time.sleep(0.1)
        pyautogui.click(board_dropdown_coordinates)
        time.sleep(0.1)

        reset_to_first_option(up_arrow_count=30)
        time.sleep(0.2)
        pyautogui.press("enter")
        time.sleep(0.2)

        # Exit the settings menu
        pyautogui.moveTo(exit_coordinates)
        time.sleep(0.1)
        pyautogui.click(exit_coordinates)
        time.sleep(0.5)  # Allow time for the board to update

        change_theme(
            settings_coordinates,
            tab_coordinates,
            piece_dropdown_coordinates,  # Use piece dropdown coordinates
            exit_coordinates,
        )