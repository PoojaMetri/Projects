import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define function to detect raised eyebrows
def detect_raised_eyebrows(landmarks, image_height):
    # Landmarks for eyebrows and eyes
    left_eyebrow = landmarks[105]
    right_eyebrow = landmarks[334]
    left_eye = landmarks[159]
    right_eye = landmarks[386]

    # Calculate vertical distance between eyebrows and eyes
    left_eyebrow_distance = abs(left_eyebrow.y - left_eye.y) * image_height
    right_eyebrow_distance = abs(right_eyebrow.y - right_eye.y) * image_height

    # Eyebrows are raised if the vertical distance is above a certain threshold
    return left_eyebrow_distance > 0.05 * image_height and right_eyebrow_distance > 0.05 * image_height

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        break

    # Flip the image horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )

            # Convert normalized landmarks to pixel coordinates
            landmarks = [
                (lm.x, lm.y) for lm in face_landmarks.landmark
            ]
            image_height, image_width, _ = frame.shape
            landmarks = [(int(x * image_width), int(y * image_height)) for x, y in landmarks]

            # Detect raised eyebrows
            if detect_raised_eyebrows(face_landmarks.landmark, image_height):
                cv2.putText(frame, "Raised Eyebrows Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Raised Eyebrows Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
