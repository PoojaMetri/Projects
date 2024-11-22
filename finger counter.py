import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands detector
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (MediaPipe works in RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get hand landmarks
        results = hands.process(frame_rgb)

        # If hand landmarks are detected, proceed to count fingers
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # List to store finger states (0 = not raised, 1 = raised)
                fingers = [0] * 5  # One entry for each finger

                # Check thumb (landmark 4 is tip, landmark 2 is base)
                if hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y:
                    fingers[0] = 1  # Thumb raised

                # Check index finger (landmark 8 is tip, landmark 6 is base)
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                    fingers[1] = 1  # Index finger raised

                # Check middle finger (landmark 12 is tip, landmark 10 is base)
                if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                    fingers[2] = 1  # Middle finger raised

                # Check ring finger (landmark 16 is tip, landmark 14 is base)
                if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                    fingers[3] = 1  # Ring finger raised

                # Check pinky finger (landmark 20 is tip, landmark 18 is base)
                if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                    fingers[4] = 1  # Pinky finger raised

                # Count the total number of raised fingers
                total_fingers = sum(fingers)

                # Display the count of raised fingers
                cv2.putText(frame, f"Fingers: {total_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame with hand landmarks and the finger count
        cv2.imshow("Finger Counter", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
