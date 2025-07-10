import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

# Finger tip landmarks
finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            fingers = []

            # --- Detect Thumb based on hand side ---
            if label == 'Right':
                fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
            else:
                fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

            # --- Detect other fingers ---
            for id in range(1, 5):
                fingers.append(
                    1 if hand_landmarks.landmark[finger_tips[id]].y < hand_landmarks.landmark[finger_tips[id] - 2].y else 0
                )

            # --- Useful landmarks for gestures ---
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            thumb_mcp = hand_landmarks.landmark[2]
            index_tip = hand_landmarks.landmark[8]
            index_dip = hand_landmarks.landmark[7]
            pinky_tip = hand_landmarks.landmark[20]
            wrist = hand_landmarks.landmark[0]

            # Normalized distance between thumb and index
            norm_dist_thumb_index = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            # --- Gesture Detection ---
            gesture = "Unknown"

            if fingers == [0, 0, 0, 0, 0]:
                gesture = "Fist"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "Open Palm"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "Victory Sign"
            elif fingers == [1, 1, 0, 0, 1]:
                gesture = "I Love You Sign"
            elif fingers == [1, 0, 0, 0, 1]:
                gesture = "Call Me Sign"
            elif fingers == [1, 0, 1, 0, 1]:
                gesture = "Rock Sign"
            elif fingers == [0, 1, 1, 1, 0]:
                gesture = "Three Finger Salute"
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "Index Pointing"
            elif fingers == [1, 0, 0, 0, 0] and thumb_tip.y < thumb_ip.y < thumb_mcp.y:
                gesture = "Thumbs Up"
            elif fingers == [1, 0, 0, 0, 0] and thumb_tip.y > thumb_ip.y > thumb_mcp.y:
                gesture = "Thumbs Down"
            elif norm_dist_thumb_index < 0.05 and index_dip.y < wrist.y:
                gesture = "OK Sign"

            # --- Display the Gesture Detected ---
            cv2.putText(frame, f'{gesture} ({label} Hand)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Accurate Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
