import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to detect faces in an image
def detect_faces(image, scaleFactor, minNeighbors, color):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    return image, faces

# Streamlit app
st.title("Face Detection using Viola-Jones Algorithm")

# Instructions
st.markdown("""
## Instructions:
1. Upload an image.
2. Adjust the parameters using the sliders.
3. Choose a color for the rectangles.
4. Click the "Detect Faces" button to detect faces.
5. Click the "Save Image" button to save the image with detected faces.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Parameters for face detection
scaleFactor = st.slider("Scale Factor", 1.01, 1.5, 1.1)
minNeighbors = st.slider("Min Neighbors", 1, 10, 5)

# Color picker for rectangle
color_hex = st.color_picker("Pick a color for the rectangles", "#00FF00")
color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert RGB to BGR

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Detect Faces'):
        # Detect faces
        result_image, faces = detect_faces(image, scaleFactor, minNeighbors, color_bgr)
        
        # Convert to RGB for displaying with Streamlit
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        st.image(result_image_rgb, caption='Image with Detected Faces', use_column_width=True)
        st.write(f"Detected {len(faces)} face(s)")
        
        # Save image with detected faces
        result_image_path = 'detected_faces.png'
        cv2.imwrite(result_image_path, result_image)
        st.write(f"Image saved as {result_image_path}")
        
        # Provide download link
        with open(result_image_path, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name="detected_faces.png",
                mime="image/png"
            )
