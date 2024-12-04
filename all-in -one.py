"""import cv2
import mediapipe as mp

# Initialize MediaPipe Hands, Face, and Pose modules
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    hand_results = hands.process(rgb_frame)
    
    # Detect pose
    pose_results = pose.process(rgb_frame)
    
    # Detect face
    face_results = face_detection.process(rgb_frame)
    
    # Draw detections on the frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Gesture logic
            thumb_is_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < \
                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
            palm_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x > \
                           hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            if thumb_is_up:
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif not thumb_is_up:
                cv2.putText(frame, "Thumbs Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if palm_is_open:
                cv2.putText(frame, "Palm Open", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Palm Closed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, "Body Detected", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)
            cv2.putText(frame, "Face Detected", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Check if face is smiling (dummy example; replace with expression model if needed)
            cv2.putText(frame, "Smile Detected", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Humanoid Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands, Face, and Pose modules
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    hand_results = hands.process(rgb_frame)

    # Detect pose
    pose_results = pose.process(rgb_frame)

    # Detect face
    face_results = face_detection.process(rgb_frame)

    # Draw detections on the frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Gesture recognition logic
            thumb_up = thumb_tip.y < thumb_ip.y
            palm_open = abs(thumb_tip.x - pinky_tip.x) > 0.2

            if thumb_up:
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Thumbs Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if palm_open:
                cv2.putText(frame, "Palm Open", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Palm Closed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, "Body Detected", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)
            cv2.putText(frame, "Face Detected", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Placeholder for smile detection (requires expression recognition model)
            cv2.putText(frame, "Smile Detected", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Humanoid Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import cv2
import mediapipe as mp

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frames with MediaPipe
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Hand detection
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check gestures
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_up = thumb_tip.y < thumb_ip.y
            palm_open = abs(thumb_tip.x - pinky_tip.x) > 0.2

            if thumb_up:
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Thumbs Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if palm_open:
                cv2.putText(frame, "Palm Open", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Palm Closed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Pose detection
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, "Body Detected", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Face detection
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

            # Placeholder for smile detection (requires facial expression model)
            cv2.putText(frame, "Smile Detected", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Humanoid Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()"""

import cv2
import mediapipe as mp

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frames with MediaPipe
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Hand detection
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check gestures
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            thumb_up = thumb_tip.y < thumb_ip.y
            palm_open = abs(thumb_tip.x - pinky_tip.x) > 0.2

            if thumb_up:
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Thumbs Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if palm_open:
                cv2.putText(frame, "Palm Open", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Palm Closed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Pose detection
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, "Body Detected", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Face detection
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

            # Placeholder for smile detection (requires facial expression model)
            cv2.putText(frame, "Smile Detected", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Humanoid Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

