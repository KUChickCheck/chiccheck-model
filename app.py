from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow all origins to make requests to your API

# Load the model
model = tf.keras.models.load_model("model_1.h5")

# Define paths for saving uploaded training images
UPLOAD_FOLDER = 'training_data'
LIVE_FOLDER = os.path.join(UPLOAD_FOLDER, 'live')
SPOOF_FOLDER = os.path.join(UPLOAD_FOLDER, 'spoof')

os.makedirs(LIVE_FOLDER, exist_ok=True)
os.makedirs(SPOOF_FOLDER, exist_ok=True)

# Image preprocessing function
def preprocess_image(image_path, target_size=(150, 150)):
    img = Image.open(image_path).convert('RGB')  # Ensure RGB format
    img = img.resize(target_size)  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Add `/predict` endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        processed_image = preprocess_image(image_file, target_size=(150, 150))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    prediction = model.predict(processed_image).tolist()[0][0]
    result = "spoof" if prediction >= 0.5 else "live"
    confidence = abs(prediction - 0.5) * 2

    return jsonify({
        'prediction': result,
        'confidence': confidence,
        'raw_output': prediction
    })

# Add `/train` endpoint
@app.route('/api/train', methods=['POST'])
def train():
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({'error': 'Image and label are required'}), 400

    image_file = request.files['image']
    label = request.form['label'].strip().lower()

    if label not in ['live', 'spoof']:
        return jsonify({'error': 'Label must be either "live" or "spoof"'}), 400

    # Save the image to the appropriate folder
    save_path = os.path.join(LIVE_FOLDER if label == 'live' else SPOOF_FOLDER, image_file.filename)
    image_file.save(save_path)

    return jsonify({'message': f'Image saved successfully under {label} category'}), 200

# Training function
def fine_tune_model():
    # Load training data
    live_images = [os.path.join(LIVE_FOLDER, img) for img in os.listdir(LIVE_FOLDER)]
    spoof_images = [os.path.join(SPOOF_FOLDER, img) for img in os.listdir(SPOOF_FOLDER)]

    images = []
    labels = []

    for img_path in live_images:
        images.append(preprocess_image(img_path, target_size=(150, 150))[0])
        labels.append(0)  # Live = 0

    for img_path in spoof_images:
        images.append(preprocess_image(img_path, target_size=(150, 150))[0])
        labels.append(1)  # Spoof = 1

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Fine-tune the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=32)

    # Save the updated model
    model.save("model_1.h5")

# Add `/retrain` endpoint
@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        fine_tune_model()
        return jsonify({'message': 'Model retrained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
