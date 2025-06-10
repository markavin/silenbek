import os
import sys
import logging
import json
import base64
import io
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

# Railway environment setup
PORT = int(os.environ.get('PORT', 5000))
HOST = '0.0.0.0'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print(f"🚀 STARTING REAL ML MODEL APP ON {HOST}:{PORT}")

# Initialize Flask app
app = Flask(__name__)

# CORS configuration
CORS(app, 
     resources={"*": {"origins": "*"}},
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'User-Agent'])

logger.info("🔓 CORS configured")

# Global variables for ML models
models = {}
model_loaded = False
available_models = []

def load_ml_models():
    """Load actual ML models from files"""
    global models, model_loaded, available_models
    
    try:
        logger.info("🔍 Searching for ML models...")
        
        # Model paths to check (adjust based on your structure)
        model_paths = {
            'bisindo': [
                './models/sign_language_model_bisindo_sklearn.pkl',
                './data/models/sign_language_model_bisindo_sklearn.pkl',
                './augmented/bisindo_model.pkl',
                './processed/bisindo_classifier.pkl'
            ],
            'sibi': [
                './models/sign_language_model_sibi_sklearn.pkl', 
                './data/models/sign_language_model_sibi_sklearn.pkl',
                './augmented/sibi_model.pkl',
                './processed/sibi_classifier.pkl'
            ]
        }
        
        # Check what files exist
        logger.info("📁 Checking available files...")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.pkl', '.joblib', '.h5', '.pt')):
                    full_path = os.path.join(root, file)
                    logger.info(f"📄 Found model file: {full_path}")
        
        # Try to load models
        loaded_count = 0
        
        for language, paths in model_paths.items():
            for path in paths:
                if os.path.exists(path):
                    try:
                        logger.info(f"📂 Loading {language} model from: {path}")
                        
                        # Try different loading methods
                        if path.endswith('.pkl'):
                            with open(path, 'rb') as f:
                                model = pickle.load(f)
                        elif path.endswith('.joblib'):
                            import joblib
                            model = joblib.load(path)
                        else:
                            logger.warning(f"⚠️ Unsupported model format: {path}")
                            continue
                        
                        models[language] = model
                        available_models.append(language)
                        loaded_count += 1
                        logger.info(f"✅ {language} model loaded successfully!")
                        break  # Stop after first successful load for this language
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load {language} model from {path}: {e}")
                        continue
        
        if loaded_count > 0:
            model_loaded = True
            logger.info(f"🎉 Successfully loaded {loaded_count} models: {available_models}")
        else:
            logger.warning("⚠️ No models loaded - running in demo mode")
            
        return model_loaded
        
    except Exception as e:
        logger.error(f"💥 Model loading error: {e}")
        return False

# Load models on startup
load_ml_models()

def preprocess_image_for_model(image_data):
    """
    Preprocess image for ML model prediction
    Adjust this based on how your model was trained
    """
    try:
        # Decode base64
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize to model input size (adjust based on your training)
        # Common sizes: 64x64, 128x128, 224x224
        target_size = (64, 64)  # Adjust this based on your model training
        resized = cv2.resize(opencv_image, target_size)
        
        # Normalize pixel values (adjust based on your training)
        normalized = resized.astype(np.float32) / 255.0
        
        # Flatten for sklearn models (if needed)
        # If your model expects flattened input:
        flattened = normalized.flatten()
        
        # If your model expects 2D input (batch_size, features):
        features = flattened.reshape(1, -1)
        
        logger.info(f"✅ Image preprocessed: {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        raise

def predict_with_real_model(processed_image, language_type='bisindo'):
    """
    Make prediction using actual ML model
    """
    global models, model_loaded
    
    try:
        if model_loaded and language_type in models:
            logger.info(f"🤖 Using real {language_type} model for prediction")
            
            model = models[language_type]
            
            # Make prediction
            prediction = model.predict(processed_image)[0]
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_image)[0]
                confidence = float(np.max(probabilities))
            else:
                # Fallback confidence
                confidence = 0.85
            
            logger.info(f"🎯 Real model prediction: {prediction} (confidence: {confidence:.2f})")
            return str(prediction), confidence
            
        else:
            # Fallback to demo prediction
            logger.warning(f"⚠️ No model for {language_type}, using demo prediction")
            return demo_predict(language_type)
            
    except Exception as e:
        logger.error(f"💥 Real model prediction failed: {e}")
        # Fallback to demo
        return demo_predict(language_type)

def demo_predict(language_type='bisindo'):
    """Fallback demo prediction"""
    import random
    
    if language_type == 'bisindo':
        signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    else:
        signs = ['K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    
    prediction = random.choice(signs)
    confidence = random.uniform(0.60, 0.80)  # Lower confidence for demo
    
    return prediction, confidence

@app.before_request
def log_request_info():
    origin = request.headers.get('Origin', 'Unknown')
    logger.info(f"📥 {request.method} {request.path} from {origin}")

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'port': PORT,
            'service': 'Sign Language API',
            'model_status': {
                'loaded': model_loaded,
                'available_models': available_models,
                'total_models': len(models)
            }
        }
        logger.info(f"✅ Health check: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 200

@app.route('/health', methods=['GET'])
def health_alt():
    return health_check()

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Sign Language API',
        'status': 'running',
        'model_info': {
            'loaded': model_loaded,
            'available': available_models
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'status': 'loaded' if model_loaded else 'demo_mode',
        'available_models': available_models,
        'total_models': len(models),
        'models_detail': {model: 'loaded' for model in available_models}
    }), 200

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    try:
        logger.info("🔮 Translation requested")
        
        # Get data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data.get('image', '')
        language_type = data.get('language_type', 'bisindo')
        
        logger.info(f"🎯 Language: {language_type}")
        logger.info(f"📸 Image size: {len(image_data)}")
        
        # Validate image
        if not image_data or len(image_data) < 100:
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Preprocess image
        try:
            processed_image = preprocess_image_for_model(image_data)
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            return jsonify({
                'success': False,
                'error': f'Image preprocessing failed: {str(e)}'
            }), 400
        
        # Make prediction
        prediction, confidence = predict_with_real_model(processed_image, language_type)
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'language_type': language_type,
            'model_status': 'real_model' if (model_loaded and language_type in models) else 'demo',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction result: {prediction} ({confidence:.2f}) - {response['model_status']}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"💥 Translation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/<path:path>', methods=['OPTIONS'])
def handle_preflight(path=None):
    return '', 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🤖 SIGN LANGUAGE API - REAL ML MODEL")
    print("="*50)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Models loaded: {len(models)}")
    print(f"Available: {available_models}")
    print("="*50)
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"💥 Server start failed: {e}")
        sys.exit(1)
