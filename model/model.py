import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('plant_disease_model.h5')

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150,3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Make a prediction
def predict(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_names = ['Healthy', 'Disease 1', 'Disease 2']  # Adjust as per your model's output classes
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Main entry point when running the script
if __name__ == "__main__":
    img_path = sys.argv[1]  # Get image path from command-line arguments
    result = predict(img_path)
    print(result)  # Output the prediction to be read by Node.js
