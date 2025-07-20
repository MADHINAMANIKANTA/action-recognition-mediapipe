import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import joblib

# Load the trained Random Forest model
model = joblib.load(r"C:\Users\mani2\Downloads\random_forest_model.pkl")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define action labels
action_labels = ["sitting", "standing", "walking", "punching", "waving"]

# Load YOLOv8 model for person detection
yolo_model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect people in the frame
    results = yolo_model(frame)

    # Iterate over each detection (multiple people)
    for result in results:
        boxes = result.boxes  # Get the detection boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])

                # Only process 'person' class (class 0 in YOLO)
                if class_id == 0:
                    person_frame = frame[y1:y2, x1:x2]
                    rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)

                    # Process the frame to extract pose landmarks
                    pose_results = pose.process(rgb_frame)

                    # Initialize an array to hold the keypoints
                    keypoints = np.zeros((33, 2))

                    # Extract keypoints if pose landmarks are detected
                    if pose_results.pose_landmarks:
                        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            keypoints[i] = [landmark.x, landmark.y]

                        # Flatten keypoints to a 1D array
                        flat_keypoints = keypoints.flatten()

                        # Ensure the number of features matches the model's expected input size
                        num_features_required = 99
                        if len(flat_keypoints) < num_features_required:
                            flat_keypoints = np.pad(flat_keypoints, (0, num_features_required - len(flat_keypoints)), mode='constant')
                        elif len(flat_keypoints) > num_features_required:
                            flat_keypoints = flat_keypoints[:num_features_required]

                        # Predict the action
                        predicted_action = model.predict([flat_keypoints])[0]

                        # Display the predicted action on the video frame
                        cv2.putText(frame, f"Action: {predicted_action}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Draw pose landmarks (keypoints) and connections (skeleton)
                        mp_drawing = mp.solutions.drawing_utils
                        mp_drawing.draw_landmarks(
                            person_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )

                    # Replace the region of interest (person's frame) with the processed result
                    frame[y1:y2, x1:x2] = person_frame

    # Show the frame
    cv2.imshow("Pose Estimation and Action Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
