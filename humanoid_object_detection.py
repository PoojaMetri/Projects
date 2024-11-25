import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is a small, pre-trained YOLOv5 model

# Define humanoid actions based on detected objects
def humanoid_action(detections):
    detected_classes = detections['name']
    if 'person' in detected_classes:
        print("Humanoid Robot Action: Greeting detected person!")
    if 'bottle' in detected_classes:
        print("Humanoid Robot Action: A bottle is detected!")
    if 'chair' in detected_classes:
        print("Humanoid Robot Action: Suggesting someone to sit.")
    # Add more actions as needed based on detected classes

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use index 0 for default webcam, 1 for an external camera

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Starting real-time object detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the webcam.")
        break

    # Perform YOLOv5 inference
    results = model(frame)

    # Parse results
    detections = results.pandas().xyxy[0]  # Extract detection details as a pandas DataFrame

    # Perform humanoid actions based on detections
    humanoid_action(detections)

    # Annotate frame with detection results
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display label and confidence
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display annotated frame
    cv2.imshow("Humanoid Object Detection", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resourcesq
cap.release()
cv2.destroyAllWindows()
