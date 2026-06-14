"""
app.py - Flask Web Application
Serves a web interface for image classification using trained model
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
try:
    model = joblib.load('savedmodel.pth')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess image for model prediction
    Converts to grayscale and resizes to match training data (64x64)
    
    Args:
        image_path: Path to image file
    
    Returns:
        flattened_image: 1D array of image features
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 64x64 (Olivetti faces standard size)
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img).flatten()
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts image file and returns predicted class
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        
        # Make prediction
        prediction = model.predict([img_array])[0]
        probabilities = model.predict_proba([img_array])[0]
        confidence = float(np.max(probabilities))
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'predicted_class': int(prediction),
            'confidence': round(confidence, 4),
            'success': True
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Return model information"""
    return jsonify({
        'model_type': 'DecisionTreeClassifier',
        'dataset': 'Olivetti Faces',
        'num_classes': 10,
        'input_shape': [1850],
        'image_size': [64, 64]
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
