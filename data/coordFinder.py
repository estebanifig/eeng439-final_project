import pyautogui

print("Move your mouse to the desired position. Press Ctrl+C to exit.")
try:
    while True:
        x, y = pyautogui.position()
        print(f"Mouse position: X={x}, Y={y}", end="\r")
except KeyboardInterrupt:
    print("\nExiting.")
