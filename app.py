import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load your Keras model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    size = (256,256)  # Resize the image to the input shape of your model
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)  # Updated from Image.ANTIALIAS to Image.Resampling.LANCZOS
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app layout
st.title("Dog vs Cat Classifier")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make the prediction
    prediction = model.predict(processed_image)
    
    # Display the result
    if prediction[0] < 0.5:
        st.write("It's a **Cat**!")
    else:
        st.write("It's a **Dog**!")
