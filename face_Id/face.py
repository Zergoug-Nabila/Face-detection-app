import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os

# Path to the Haar Cascade XML file
face_cascade_path = 'C:/Users/nabila/Desktop/GoMyCode/face_Id/haarcascade_frontalface_default streamlit_1.xml'

# Print statements to debug the file path
print("Cascade file path:", face_cascade_path)
print("File exists:", os.path.isfile(face_cascade_path))

face_cascade = cv2.CascadeClassifier(face_cascade_path)

def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def capture_frame():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to capture an image from your webcam and detect faces.")

    # Add a button to capture an image and detect faces
    if st.button("Capture Image and Detect Faces"):
        # Capture a frame from the webcam
        frame = capture_frame()
        if frame is not None:
            # Detect faces in the captured frame
            detected_frame = detect_faces(frame)
            # Convert the frame to RGB format for displaying in Streamlit
            detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image for displaying in Streamlit
            img = Image.fromarray(detected_frame_rgb)
            st.image(img, caption='Detected Faces', use_column_width=True)
        else:
            st.write("Failed to capture an image from the webcam.")

if __name__ == "__main__":
    app()
