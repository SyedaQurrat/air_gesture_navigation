import cv2
import mediapipe as mp
import numpy as np
import webbrowser
import threading
import time
import pyautogui

# Mobile Camera URL
MOBILE_CAM_URL = "mobile camera access"  # Update this

# Website Mapping with Buttons
buttons = [
    ("Google", "https://www.google.com"),
    ("Facebook", "https://www.facebook.com"),
    ("YouTube", "https://www.youtube.com"),
    ("Hekto", "http://localhost:3000"),
    ("Login", "http://localhost:3000/login"),
    ("Contact", "http://localhost:3000/contact"),
    ("About", "http://localhost:3000/about"),
    ("Shop", "http://localhost:3000/shop"),
    ("Cart", "http://localhost:3000/cart"),
    ("Wishlist", "http://localhost:3000/wishlist"),
]

# MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open Website Function
last_clicked_time = 0
click_delay = 5  # 3 second delay
last_opened_tab = None  # Store the last opened tab

def open_website(url):
    global last_clicked_time, last_opened_tab
    current_time = time.time()
    if current_time - last_clicked_time > click_delay:
        last_clicked_time = current_time
        last_opened_tab = url
        threading.Thread(target=webbrowser.open_new_tab, args=(url,)).start()

# Close Last Opened Tab
def close_last_tab():
    global last_opened_tab
    if last_opened_tab:
        pyautogui.hotkey("ctrl", "w")  # Close current tab
        last_opened_tab = None

# Open Mobile Camera
cap = cv2.VideoCapture(MOBILE_CAM_URL)

if not cap.isOpened():
    print("Error: Unable to access mobile webcam. Check the IP address and ensure DroidCam/IP Webcam is running.")
    exit()

cursor_position = None
writing = False

# Button Positions
button_x = 50
button_y_start = 100
button_height = 60
button_width = 200
gap = 10

# Close Button Position
close_button_x = 300
close_button_y = 100
close_button_width = 150
close_button_height = 60

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to fetch frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert coordinates to screen space
            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            cursor_position = (x, y)

            # Detect if the fist is closed
            distance = np.linalg.norm(
                np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            )

            if distance > 0.1:  # If fingers are apart → Writing Mode
                writing = True
            else:  # If fingers are close → Stop Writing
                writing = False

    # Draw Cursor
    if cursor_position:
        color = (0, 255, 0) if writing else (0, 0, 255)  # Green for writing, Red for stopped
        cv2.circle(frame, cursor_position, 10, color, -1)

    # Draw Buttons
    for i, (text, url) in enumerate(buttons):
        button_y = button_y_start + i * (button_height + gap)
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), (0, 0, 255), -1)
        cv2.putText(frame, text, (button_x + 20, button_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check Cursor Overlap with Button
        if cursor_position and button_x < cursor_position[0] < button_x + button_width and button_y < cursor_position[1] < button_y + button_height:
            open_website(url)

    # Draw Close Button
    cv2.rectangle(frame, (close_button_x, close_button_y), 
                  (close_button_x + close_button_width, close_button_y + close_button_height), (255, 0, 0), -1)
    cv2.putText(frame, "Close Tab", (close_button_x + 20, close_button_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Check Cursor Overlap with Close Button
    if cursor_position and close_button_x < cursor_position[0] < close_button_x + close_button_width and close_button_y < cursor_position[1] < close_button_y + close_button_height:
        close_last_tab()

    cv2.imshow("Hand Gesture Navigation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
