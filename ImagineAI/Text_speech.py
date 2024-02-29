import cv2
from PIL import Image
import pytesseract
import pyttsx3

def tesseract(image):
    path_to_tesseract = r"E:\Tesseract\tesseract.exe"  
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(image)
    return text

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    camera = cv2.VideoCapture(0)

    while True:
        _, img = camera.read()
        
        # Perform text detection
        text = tesseract(Image.fromarray(img))
        print("Detected Text:", text)
        
        # Speak the detected text
        speak_text(text)
        
        # Display the image with detected text
        cv2.imshow('Text detection', img)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
