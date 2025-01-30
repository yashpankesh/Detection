import cv2
import face_recognition
import numpy as np
import os
from ultralytics import YOLO  # Import YOLO

# Load YOLOv8 Model
model = YOLO(r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\models\yolov8s.pt")

# Face Recognition Setup
known_face_encodings = []
known_face_names = []

def load_face_encoding(image_path):
    """Load face encoding from an image file if it exists."""
    if not os.path.exists(image_path):
        print(f"Error: File not found -> {image_path}")
        return None
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# Known faces
known_faces = {
    "Yash": r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\Detection\images\yp.jpg",
    "Khushbu": r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\Detection\images\_MG_7750.JPG",
    "Person3": r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\Detection\images\PXL_20241102_062005971.PORTRAIT.jpg",
    "Person4": r"C:\Users\ptlya\OneDrive\Desktop\Python\detection\Detection\images\IMG_7364.JPG"
}

# Load encodings
for name, image_path in known_faces.items():
    encoding = load_face_encoding(image_path)
    if encoding is not None:
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    else:
        print(f"Warning: No face found in {image_path}. Skipping.")

# Use IP Webcam
ip_webcam_url = "http://10.61.88.10:8080/video"  # Replace with your actual IP
video_capture = cv2.VideoCapture(ip_webcam_url)

# Set the desired resolution (width, height)
desired_width = 640  # Change this value as per your requirement
desired_height = 480  # Change this value as per your requirement
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize frame (if necessary, in case webcam resolution is different from desired)
    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    # Face Recognition
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        top, right, bottom, left = face_location
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if len(matches) > 0 and any(matches):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw face box
        cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame_resized, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Object Detection with YOLOv8
    results = model(frame_resized)  # Run YOLOv8 inference

    # Draw bounding boxes for detected objects
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            label = model.names[int(box.cls[0])]  # Object label

            if conf > 0.3:  # Confidence threshold
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_resized, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display frame
    cv2.imshow("Face and Object Recognition (YOLOv8)", frame_resized)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
