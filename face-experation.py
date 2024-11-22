import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define functions to detect facial expressions
def detect_smile(landmarks, image_height):
    # Landmarks for mouth corners and lips
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[78]
    right_mouth = landmarks[308]

    # Calculate the vertical distance between top and bottom lips
    lip_distance = abs(top_lip.y - bottom_lip.y) * image_height

    # Calculate horizontal mouth width
    mouth_width = abs(left_mouth.x - right_mouth.x) * image_height

    # Smiling if lip distance is significant compared to mouth width
    return lip_distance > 0.2 * mouth_width

def detect_raised_eyebrows(landmarks, image_height):
    # Landmarks for eyebrows and eyes
    left_eyebrow = landmarks[105]
    left_eye = landmarks[159]
    right_eyebrow = landmarks[334]
    right_eye = landmarks[386]

    # Calculate vertical distance between eyebrows and eyes
    left_eyebrow_distance = abs(left_eyebrow.y - left_eye.y) * image_height
    right_eyebrow_distance = abs(right_eyebrow.y - right_eye.y) * image_height

    # Eyebrows are raised if distances are above a threshold
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

            # Detect expressions
            if detect_smile(face_landmarks.landmark, image_height):
                cv2.putText(frame, "Smile Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if detect_raised_eyebrows(face_landmarks.landmark, image_height):
                cv2.putText(frame, "Raised Eyebrows Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
