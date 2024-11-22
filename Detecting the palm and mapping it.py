# detecting palm and map it:
import cv2 #card varification value
import mediapipe as mp

# Initialize Mediapipe's hand detection and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)#only 1 palm

# Open the webcam (default camera is usually index 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():# return true if the camera is open else return false
    success, frame = cap.read()# is open true than read the frame 
    if not success: # if the cap is not open than not success
        print("Failed to grab frame")
        break # if not success than break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)# like a mirror u can see yourself

    # Convert BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# red,gree,blue=color # in opencv it is BRG colour we have to connevrt it to RGB color

    # Process the frame with MediaPipe to detect hands
    result = hands.process(rgb_frame)
    # handmark=mapping on the hand making the points on the hand.
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
#lm=landmark
            # Print and display the landmarks' (x, y) coordinates on the frame
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape # h is height and w is width
                cx, cy = int(lm.x * w), int(lm.y * h) # it will give correct pixel value and drawn on frame
                cv2.putText(frame, f"{idx}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Display the output frame
    cv2.imshow("Palm Detection and Mapping", frame)

    # Exit when the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27: #5 miliseconds for pressing key and 27 is a ACSII value of ESC 
        break

# Release resources
cap.release() # just release the camara 
cv2.destroyAllWindows() # close all thw window