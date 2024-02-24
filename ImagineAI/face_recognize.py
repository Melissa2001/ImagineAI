import cv2
import os
import numpy as np
import pyttsx3 
# Use LBPH Face Recognizer
model = cv2.face.LBPHFaceRecognizer_create() #cv2.face.FisherFaceRecognizer_create()
#EigenFaceRecognizer,model = cv2.face.LBPHFaceRecognizer_create()

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

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

        confidence = prediction[1]  # Access confidence value directly
        if confidence < 500:  # Adjust threshold as needed

            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], confidence),
                        (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            person_name = names[prediction[0]]  # Get the person's name
            cv2.putText(im, f'{person_name}- {confidence:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # Say the person's name using text-to-speech
            engine = pyttsx3.init()  # Initialize the text-to-speech engine
            engine.say(person_name+" is infront of u")  # Speak the person's name
            engine.runAndWait() 
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            engine = pyttsx3.init()  # Initialize the text-to-speech engine
            engine.say("Some one is infront of you")  
            engine.runAndWait() 
    cv2.imshow('OpenCV', im)

    # Check for the 'Esc' key press
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()