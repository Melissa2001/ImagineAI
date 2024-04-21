import cv2
import numpy as np
import math
import time
import pyttsx3  # Library for text-to-speech conversion

from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")

# Variable to store the timestamp of the last label
last_label_time = time.time()

# Define the Region of Interest (ROI) coordinates for three regions
roi_coordinates1 = (10, 50, 250, 500)  # Red ROI
roi_coordinates2 = (275, 50, 250, 500)  # Green ROI
roi_coordinates3 = (540, 50, 250, 500)  # Blue ROI

# Load the laptop camera
cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Crop the frame to the specified ROIs
    for roi_coordinates, color in zip(
        [roi_coordinates1, roi_coordinates2, roi_coordinates3],
        [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
    ):
        x, y, w, h = roi_coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        roi_frame = frame[y : y + h, x : x + w]

        # Predict with the model using the cropped ROI frame
        results = model(
            source=roi_frame, show=False, conf=0.4, verbose=False, save=False
        )

        detected = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Calculate distance
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance >= 6.0 and conf > 0.4:
                    detected = True
                    break

        # Take actions based on detected objects in each region
        if detected:
            if color == (0, 0, 255):
                direction = "Move left"
            elif color == (0, 255, 0):
                direction = "Move right"
            else:
                direction = "Move left"

            # Output direction via audio
            engine.say(direction)
            engine.runAndWait()

    # Display the frame with colored ROIs
    cv2.imshow("Camera Feed with Colored ROIs", frame)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
