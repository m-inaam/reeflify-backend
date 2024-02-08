from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

TF_MODEL_FILE_PATH = 'model.tflite'

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
interpreter.allocate_tensors()

# Define class names
class_names = ['bleached', 'dead', 'healthy']

# Specify image dimensions
img_width, img_height = 180, 180

# Set up the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'})

        file = request.files['image']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read the image and perform inference
            img = Image.open(file_path)
            img = img.resize((img_width, img_height))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32)

            # Run inference with TensorFlow Lite model
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
            interpreter.invoke()
            output_array = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

            # Get prediction results
            prediction_lite = np.argmax(output_array)
            confidence_lite = 100 * np.max(tf.nn.softmax(output_array))

            # Return the prediction as JSON
            return jsonify({'prediction': class_names[prediction_lite], 'confidence': confidence_lite})

        else:
            return jsonify({'error': 'Invalid file format'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
