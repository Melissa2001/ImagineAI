import cv2
import numpy as np
import pyttsx3
import os
import sys
import torch
import time
from pathlib import Path
import speech_recognition as sr
from PIL import Image
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Path to YOLOv5 weights
weights_path = r"C:\Users\CHRISTINE\Documents\KARUN\programs\ImagineAI\ImagineAI\yolov5s.pt"  # Update with the path to the downloaded YOLOv5 weights

# Load the pre-trained YOLOv5 object detection model
sys.path.append(str(Path(weights_path).parents[0]))  # add yolov5/ to path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load YOLOv5 model
device = select_device('')
model = attempt_load(weights_path)  # load FP32 model
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

# Function to detect objects in a frame and convert the results to speech
def detect_objects(frame):
    global last_recognition_time
    global recognized_objects  # Declare recognized_objects as global

    # Initialize recognized_objects if not defined
    if 'recognized_objects' not in globals():
        recognized_objects = set()

    # Define cooldown time
    cooldown_time = 15  # in seconds

    # Check if cooldown time has elapsed
    if 'last_recognition_time' not in globals() or time.time() - last_recognition_time >= cooldown_time:
        recognized_objects.clear()
        last_recognition_time = time.time()

    # Resize frame to the model input size
    img = cv2.resize(frame, (640, 640))
    img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Convert image to torch tensor
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Predict
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):
        # If detections exist
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Loop through detections and draw bounding boxes
            for *xyxy, conf, cls in reversed(det):
                class_name = names[int(cls)]
                label = f'{class_name}'

                # Check if the object has been recognized recently
                if class_name not in recognized_objects:
                    recognized_objects.add(class_name)

                    # Convert class name to speech
                    engine.say(class_name)
                    engine.runAndWait()

                    # Update last recognition time
                    last_recognition_time = time.time()

                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Function for facial recognition
def facial_recognition():
    # Initialize necessary variables
    persons_in_frame = []
    last_recognition_time = {}

    # Use LBPH Face Recognizer
    model = cv2.face.LBPHFaceRecognizer_create()

    size = 4
    haar_file = 'ImagineAI\haarcascade_frontalface_default.xml'
    datasets = 'imagineAI\datasets'

    # Part 1: Create a face recognizer
    print('Recognizing Face. Please be in sufficient light...')

    # Create lists for images and corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = os.path.join(subjectpath, filename)
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1

    (width, height) = (130, 100)

    # Create a Numpy array from the two lists
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # Train the face recognizer
    model.train(images, labels)

    # Part 2: Use the recognizer on a camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)

    while True:
        (status, im) = webcam.read()  # Read from camera

        if not status:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

            # Recognize the face using the trained model
            prediction = model.predict(face_resize)

            # Display prediction information and bounding box
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            person_id = prediction[0]
            confidence = prediction[1]

            if confidence < 500:  # Adjust threshold as needed
                person_name = names.get(person_id, "Unknown")

                # Check if the person has been recognized recently
                if person_id not in persons_in_frame:
                    cv2.putText(im, f'{person_name}- {confidence:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                    if person_name != "Unknown":
                        # Say the person's name using text-to-speech
                        engine.say(person_name + " is in front of you")  # Speak the person's name
                        engine.runAndWait()

                    # Update last recognition time for this person
                    last_recognition_time[person_id] = time.time()
                    persons_in_frame.append(person_id)

        # Remove persons who are no longer in the frame
        persons_in_frame = [person_id for person_id in persons_in_frame if time.time() - last_recognition_time.get(person_id, 0) < 5]

        cv2.imshow('OpenCV', im)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()


# Function to save a new face
def save_new_face():
    datasets = 'imagineAI\datasets'
    face_cascade = cv2.CascadeClassifier('ImagineAI\haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)
    width, height = 130, 100

    # Get the name of the new person through audio input
    engine.say("Please say the name of the new person.")
    engine.runAndWait()

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak:")
        audio = recognizer.listen(source)

    try:
        name = recognizer.recognize_google(audio)
        print("You said:", name)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return

    # Create directory for the new person
    path = os.path.join(datasets, name)
    if not os.path.isdir(path):
        os.mkdir(path)

    count = 1
    while count < 30:
        _, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('% s/% s.png' % (path, count), face_resize)
            count += 1

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

# Function for text recognition
def text_recognition():
    camera = cv2.VideoCapture(0)

    while True:
        _, img = camera.read()
        
        # Perform text detection
        text = pytesseract.image_to_string(img)
        print("Detected Text:", text)

        # Speak the detected text
        engine.say(text)
        engine.runAndWait()

        # Display the image with detected text
        cv2.imshow('Text detection', img)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Execute different functions based on voice commands
def execute_command(command):
    if command == "object":
        detect_objects()
    elif command == "facial":
        facial_recognition()
    elif command == "detection":
        save_new_face()
    elif command == "scanning":
        text_recognition()
    elif command == "exit":
        sys.exit()  # Exit the program

# Voice command recognition
def recognize_voice_command():
    # Initialize recognizer instance
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening for a voice command...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise

        try:
            # Capture audio data from the microphone
            audio_data = recognizer.listen(source, timeout=10)  # Listen for up to 5 seconds for a command

            # Recognize speech using Google Speech Recognition
            command = recognizer.recognize_google(audio_data)

            print("Recognized command:", command)

            return command.lower()

        except sr.WaitTimeoutError:
            print("No voice command detected within the timeout period.")
            return ""
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print("Error accessing the Google Speech Recognition API:", e)
            return ""

# Main function
# Inside the main function
def main():
    while True:
        # Recognize voice commands
        command = recognize_voice_command()

        # Execute the corresponding action based on the voice command
        if command == "object":
            # Capture a frame from the camera
            _, frame = cv2.VideoCapture(0).read()
            # Call detect_objects with the captured frame
            detect_objects(frame)
        else:
            execute_command(command)
if __name__ == "__main__":
    main()
