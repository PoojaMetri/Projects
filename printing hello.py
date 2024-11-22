import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check for open-palm gesture based on landmarks
                # (landmark 8 - tip of index finger, landmark 4 - tip of thumb)
                index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                # Determine open or closed palm by relative positions
                if index_tip_y < thumb_tip_y:
                    print("Hello")
                else:
                    print("", end="")  # Closed palm, print nothing

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
