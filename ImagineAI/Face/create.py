import cv2
import os
import numpy as np
import pyttsx3

# Function to train the face recognizer model
def train_model():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create lists for images and corresponding labels
    images = []
    labels = []
    names = {}

    for idx, name in enumerate(os.listdir('datasets')):
        names[idx] = name
        for image_name in os.listdir(os.path.join('datasets', name)):
            img = cv2.imread(os.path.join('datasets', name, image_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))  # Resize the image to a fixed size
            images.append(img)
            labels.append(idx)

    images = np.array(images)
    labels = np.array(labels)

    # Train the face recognizer model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

    return recognizer, names


# Function to recognize faces in real-time
def recognize_faces(recognizer, names):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    while True:
        _, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            # Recognize the face using the trained model
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 100:
                cv2.putText(frame, names[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Speak the recognized person's name
                engine.say(names[label])
                engine.runAndWait()
            else:
                cv2.putText(frame, 'Unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Recognize Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    engine.shutdown()

if __name__ == "__main__":
    # Train the face recognizer model
    recognizer, names = train_model()

    # Recognize faces in real-time
    recognize_faces(recognizer, names)
