import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the thumbs-up detection logic
def is_thumbs_up(hand_landmarks):
    # Thumb tip (landmark 4) and MCP (landmark 2)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    # Index finger tip (landmark 8)
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Check if thumb tip is above the MCP (extended upwards) 
    # and index finger is below the MCP (closed fist except thumb)
    if thumb_tip.y < thumb_mcp.y and index_tip.y > thumb_mcp.y:
        return True
    return False

# Start processing video feed
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process the frame to detect hands
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Draw the hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for thumbs-up gesture
                if is_thumbs_up(hand_landmarks):
                    cv2.putText(frame, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display the output
        cv2.imshow('Thumbs-Up Detection', frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
