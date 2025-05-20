from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('rare_disease_cnn_model.h5')

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    img = Image.open(file)
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    return f"Predicted disease class: {predicted_class[0]}"

if __name__ == "__main__":
    app.run(debug=True)
