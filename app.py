import cv2
import mediapipe as mp
import numpy as np

# Mobile Camera URL (Update your phone's IP)
MOBILE_CAM_URL = "mobile camera access"

# Colors & Tools
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
current_color = colors[0]  # Default Blue
tool = "pencil"  # Default tool

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start capturing video
cap = cv2.VideoCapture(MOBILE_CAM_URL)
if not cap.isOpened():
    print("Error: Unable to access mobile webcam.")
    exit()

drawn_points = []  # Store drawn points

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to fetch frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    cursor_position = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert to screen coordinates
            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            cursor_position = (x, y)

            # Detect if fingers are touching (selection mode)
            distance = np.linalg.norm(
                np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            )

            # **Check if selecting a color**
            if distance < 0.03:  # Reduced for better selection
                if 10 <= x <= 60 and 10 <= y <= 50:
                    current_color = colors[0]  # Blue
                elif 70 <= x <= 120 and 10 <= y <= 50:
                    current_color = colors[1]  # Green
                elif 130 <= x <= 180 and 10 <= y <= 50:
                    current_color = colors[2]  # Red
                elif 190 <= x <= 240 and 10 <= y <= 50:
                    current_color = colors[3]  # Yellow

                # **Check if selecting a tool**
                elif frame.shape[1] - 70 <= x <= frame.shape[1] - 20 and 100 <= y <= 150:
                    tool = "pencil"
                elif frame.shape[1] - 70 <= x <= frame.shape[1] - 20 and 160 <= y <= 210:
                    tool = "eraser"

            if distance > 0.07:  # Fingers apart → Writing Mode
                if tool == "pencil":
                    drawn_points.append((cursor_position, current_color))
                elif tool == "eraser":
                    drawn_points = []  # Erase everything

    # Draw Writing Path
    for i in range(1, len(drawn_points)):
        if drawn_points[i] and drawn_points[i - 1]:
            cv2.line(frame, drawn_points[i - 1][0], drawn_points[i][0], drawn_points[i][1], 3)

    # Draw Cursor
    if cursor_position:
        cv2.circle(frame, cursor_position, 10, current_color, -1)

    # Draw Color Selection Boxes (Top Left)
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (10 + i * 60, 10), (60 + i * 60, 50), color, -1)
        if current_color == color:
            cv2.rectangle(frame, (10 + i * 60, 10), (60 + i * 60, 50), (255, 255, 255), 2)

    # Draw Tool Selection (Right Side)
    cv2.rectangle(frame, (frame.shape[1] - 70, 100), (frame.shape[1] - 20, 150), (255, 255, 255), -1)
    cv2.putText(frame, "✏️", (frame.shape[1] - 60, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(frame, (frame.shape[1] - 70, 160), (frame.shape[1] - 20, 210), (200, 200, 200), -1)
    cv2.putText(frame, "🧽", (frame.shape[1] - 60, 195), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Hand Gesture Drawing", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






# # is code me butoon kaie bar click ho rhy hain
# import cv2
# import mediapipe as mp
# import numpy as np
# import webbrowser
# import threading

# # Mobile Camera URL (Replace with actual phone's IP)
# MOBILE_CAM_URL = "mobile camera access"  # Update this

# # Website Mapping with Buttons
# buttons = [
#     ("Google", "https://www.google.com"),
#     ("Facebook", "https://www.facebook.com"),
#     ("YouTube", "https://www.youtube.com"),
#     ("Hekto", "http://localhost:3000"),
#     ("Login", "http://localhost:3000/login"),
#     ("Contact", "http://localhost:3000/contact"),
#     ("About", "http://localhost:3000/about"),
#     ("Shop", "http://localhost:3000/shop"),
#     ("Cart", "http://localhost:3000/cart"),
#     ("Wishlist", "http://localhost:3000/wishlist"),
# ]

# # MediaPipe Hand Tracking
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# # Open Website Function
# def open_website(url):
#     threading.Thread(target=webbrowser.open_new_tab, args=(url,)).start()

# # Open Mobile Camera
# cap = cv2.VideoCapture(MOBILE_CAM_URL)

# if not cap.isOpened():
#     print("Error: Unable to access mobile webcam. Check the IP address and ensure DroidCam/IP Webcam is running.")
#     exit()

# cursor_position = None
# writing = False

# # Button Positions
# button_x = 50
# button_y_start = 100
# button_height = 60
# button_width = 200
# gap = 10

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Unable to fetch frame from webcam.")
#         break

#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             index_tip = hand_landmarks.landmark[8]
#             thumb_tip = hand_landmarks.landmark[4]

#             # Convert coordinates to screen space
#             x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
#             cursor_position = (x, y)

#             # Detect if the fist is closed
#             distance = np.linalg.norm(
#                 np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
#             )

#             if distance > 0.1:  # If fingers are apart → Writing Mode
#                 writing = True
#             else:  # If fingers are close → Stop Writing
#                 writing = False

#     # Draw Cursor
#     if cursor_position:
#         color = (0, 255, 0) if writing else (0, 0, 255)  # Green for writing, Red for stopped
#         cv2.circle(frame, cursor_position, 10, color, -1)

#     # Draw Buttons
#     for i, (text, url) in enumerate(buttons):
#         button_y = button_y_start + i * (button_height + gap)
#         cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), (0, 0, 255), -1)
#         cv2.putText(frame, text, (button_x + 20, button_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Check Cursor Overlap with Button
#         if cursor_position and button_x < cursor_position[0] < button_x + button_width and button_y < cursor_position[1] < button_y + button_height:
#             open_website(url)

#     cv2.imshow("Hand Gesture Navigation", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()