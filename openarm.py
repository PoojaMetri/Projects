# open arms
import cv2  # card verification value cv2
import mediapipe as mp  # framework for vision data such as video or audio
import math

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start the webcam capture
cap = cv2.VideoCapture(0)  # cap = video capture

# Initialize the pose detector
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()  # ret=regular expression tool
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        # Draw landmarks on the original frame for visualization
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if result.pose_landmarks:
            # Extract key landmarks (e.g., wrists, shoulders)
            landmarks = result.pose_landmarks.landmark

            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Convert normalized coordinates to pixel values
            frame_height, frame_width, _ = frame.shape
            lw_x, lw_y = int(left_wrist.x * frame_width), int(left_wrist.y * frame_height)
            rw_x, rw_y = int(right_wrist.x * frame_width), int(right_wrist.y * frame_height)

            ls_x, ls_y = int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height)
            rs_x, rs_y = int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)

            # Calculate the distance between hands (open arms check)
            hand_distance = math.sqrt((rw_x - lw_x) ** 2 + (rw_y - lw_y) ** 2)
            shoulder_distance = math.sqrt((rs_x - ls_x) ** 2 + (rs_y - ls_y) ** 2)

            # Detect "Open Arms" if hands are far apart from each other and shoulders
            if hand_distance > 1.5 * shoulder_distance:  # Added 'if' here
                cv2.putText(frame, "Open Arms Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the output frame
        cv2.imshow('Open Pose Detection', frame)

        # Using a more reasonable wait time
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
