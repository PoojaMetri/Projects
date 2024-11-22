import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to determine the gesture based on the hand landmarks
def get_gesture(hand_landmarks):
    # We will use the position of specific fingers to determine the gesture
    # Example: Simple checks for rock, paper, and scissors gestures
    
    # Rock: fist (closed hand)
    # Paper: open hand
    # Scissors: index and middle finger extended

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Check if all fingers are curled (Rock)
    if thumb_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y and \
       index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
       middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
       ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
       pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        return 'Rock'
    
    # Check if all fingers are extended (Paper)
    if index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
       middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
       ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
       pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        return 'Paper'
    
    # Check if index and middle finger are extended (Scissors)
    if index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
       middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
       ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
       pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
        return 'Scissors'
    
    return None

# Function to determine the winner
def determine_winner(user_choice, robot_choice):
    if user_choice == robot_choice:
        return "It's a tie!"
    elif (user_choice == 'Rock' and robot_choice == 'Scissors') or \
         (user_choice == 'Paper' and robot_choice == 'Rock') or \
         (user_choice == 'Scissors' and robot_choice == 'Paper'):
        return "You win!"
    else:
        return "Robot wins!"

# Start processing video feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        break

    # Flip the image horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    # Process the frame to detect hand landmarks
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # Draw the hand annotations on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the user's hand gesture
            user_gesture = get_gesture(hand_landmarks)

            # If a gesture is detected, simulate robot's move and determine the winner
            if user_gesture:
                # Simulate the robot's random choice
                robot_gesture = random.choice(['Rock', 'Paper', 'Scissors'])

                # Determine the winner
                result = determine_winner(user_gesture, robot_gesture)

                # Display the results
                cv2.putText(frame, f"Your move: {user_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Robot's move: {robot_gesture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Result: {result}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Rock Paper Scissors', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
