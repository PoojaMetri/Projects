import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Drawing modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Detection and Face Mesh models
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image color from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face detection
        face_results = face_detection.process(frame_rgb)

        # Process the frame for face mesh (facial landmarks)
        face_mesh_results = face_mesh.process(frame_rgb)

        # Draw the bounding box around the detected face
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw a square bounding box around the face
                side_length = max(w, h)
                x_center = x + w // 2
                y_center = y + h // 2
                x = x_center - side_length // 2
                y = y_center - side_length // 2

                cv2.rectangle(frame, (x, y), (x + side_length, y + side_length), (0, 255, 0), 2)

        # Draw the facial landmarks if face mesh is found
        if face_mesh_results.multi_face_landmarks:
            for landmarks in face_mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

        # Display the resulting frame
        cv2.imshow('Face Detection with Square Bounding Box and Landmarks', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
