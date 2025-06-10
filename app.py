import os
import sys
import cv2
import numpy as np
import base64
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime
import io
from PIL import Image
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Railway environment setup - CRITICAL: Get port from environment
PORT = int(os.environ.get('PORT', 5000))
HOST = '0.0.0.0'

# Setup logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Log startup info
logger.info(f"Starting Flask app on {HOST}:{PORT}")
logger.info(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")

# Initialize Flask app FIRST - CRITICAL for Railway
app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})

# Add hostname validation for Railway healthchecks
@app.before_request
def check_hostname():
    """Allow Railway healthcheck hostname"""
    allowed_hosts = [
        'healthcheck.railway.app',  # Railway healthcheck domain
        'localhost',
        '127.0.0.1',
        '0.0.0.0'
    ]
    
    if request.host:
        host = request.host.split(':')[0]  # Remove port
        if any(allowed in request.host for allowed in allowed_hosts):
            return None
        # Allow any Railway domain
        if '.railway.app' in request.host or '.up.railway.app' in request.host:
            return None
    
    return None  # Allow all for now

# IMMEDIATE health check that doesn't depend on ML models
@app.route('/api/health', methods=['GET'])
def health_check():
    """Lightweight health check for Railway - loads instantly"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'port': PORT,
            'host': HOST,
            'python_version': sys.version,
            'flask_ready': True
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ROOT ENDPOINT for Railway
@app.route('/', methods=['GET'])
def root():
    """Root endpoint for Railway"""
    return jsonify({
        'service': 'Sign Language API',
        'status': 'running',
        'endpoints': ['/api/health', '/api/translate', '/api/models'],
        'timestamp': datetime.now().isoformat()
    })

# Global variable for API instance
api = None

def init_ml_models():
    """Initialize ML models in background - don't block startup"""
    global api
    try:
        logger.info("Loading heavy dependencies...")
        import mediapipe as mp
        import pickle
        import joblib
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_file_dir

        if project_root not in sys.path:
            sys.path.append(project_root)

        # Try to import feature extractor
        try:
            from src.data_preprocessing.feature_extractor import extract_features
            extract_features_available = True
            logger.info("Feature extractor imported successfully")
        except ImportError as e:
            extract_features_available = False
            logger.warning(f"Feature extractor not available: {e}")

        logger.info("Heavy dependencies loaded successfully")
        
        # Initialize API class
        api = EnhancedSignLanguageAPI()
        logger.info("ML API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to load ML dependencies: {e}")
        logger.error(traceback.format_exc())

class EnhancedSignLanguageAPI:
    def __init__(self):
        self.models = {}
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = current_file_dir
        self.mp_hands = None
        self.hands = None
        
        try:
            # Enhanced MediaPipe setup for better two-hand detection
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,  # Ensure 2 hands
                min_detection_confidence=0.5,  # Lower threshold for better detection
                min_tracking_confidence=0.4    # Lower threshold for tracking
            )
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")
        
        # Load models in background
        try:
            self.load_models()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            
    def load_models(self):
        """Load available models with error handling"""
        logger.info("Loading models...")
        
        # Check if models directory exists
        models_dir = os.path.join(self.project_root, 'data', 'models')
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return
        
        model_configs = [
            {
                'name': 'SIBI',
                'sklearn_path': os.path.join(models_dir, 'sign_language_model_sibi_sklearn.pkl'),
                'tensorflow_path': os.path.join(models_dir, 'sign_language_model_sibi_tensorflow.h5'),
                'tensorflow_meta_path': os.path.join(models_dir, 'sign_language_model_sibi_tensorflow_meta.pkl'),
            },
            {
                'name': 'BISINDO',
                'sklearn_path': os.path.join(models_dir, 'sign_language_model_bisindo_sklearn.pkl'),
                'tensorflow_path': os.path.join(models_dir, 'sign_language_model_bisindo_tensorflow.h5'),
                'tensorflow_meta_path': os.path.join(models_dir, 'sign_language_model_bisindo_tensorflow_meta.pkl'),
            }
        ]
        
        for config in model_configs:
            try:
                model_info = {'available_models': []}
                
                # Load sklearn model (prefer this for stability)
                if os.path.exists(config['sklearn_path']):
                    try:
                        import joblib
                        sklearn_data = joblib.load(config['sklearn_path'])
                        if self.validate_sklearn_model(sklearn_data, config['name']):
                            model_info['sklearn_model'] = sklearn_data
                            model_info['available_models'].append('sklearn')
                            logger.info(f"  {config['name']}: Scikit-learn model loaded")
                    except Exception as e:
                        logger.warning(f"  {config['name']}: Scikit-learn load failed - {e}")
                
                # Load TensorFlow model
                if os.path.exists(config['tensorflow_path']) and os.path.exists(config['tensorflow_meta_path']):
                    try:
                        import tensorflow as tf
                        import joblib
                        tf_model = tf.keras.models.load_model(config['tensorflow_path'])
                        tf_meta = joblib.load(config['tensorflow_meta_path'])
                        
                        if self.validate_tensorflow_model(tf_model, tf_meta, config['name']):
                            model_info['tensorflow_model'] = tf_model
                            model_info['tensorflow_meta'] = tf_meta
                            model_info['available_models'].append('tensorflow')
                            logger.info(f"  {config['name']}: TensorFlow model loaded")
                    except Exception as e:
                        logger.warning(f"  {config['name']}: TensorFlow load failed - {e}")
                
                if model_info['available_models']:
                    self.models[config['name']] = model_info
                    
            except Exception as e:
                logger.error(f"Failed to load {config['name']}: {e}")
        
        logger.info(f"Loaded {len(self.models)} language models: {list(self.models.keys())}")
    
    def validate_sklearn_model(self, model_data, language):
        """Validate sklearn model"""
        try:
            model = model_data['model']
            test_data = np.random.rand(1, getattr(model, 'n_features_in_', 50)) * 0.1
            pred = model.predict(test_data)[0]
            return True
        except Exception as e:
            logger.error(f"{language} sklearn validation failed: {e}")
            return False
    
    def validate_tensorflow_model(self, model, meta, language):
        """Validate TensorFlow model"""
        try:
            input_shape = model.input_shape[1]
            test_data = np.random.rand(1, input_shape) * 0.1
            pred_prob = model.predict(test_data, verbose=0)
            return True
        except Exception as e:
            logger.error(f"{language} TensorFlow validation failed: {e}")
            return False

# API status endpoint that checks if ML models are loaded
@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        if not api:
            return jsonify({
                'available_models': [],
                'total_models': 0,
                'status': 'models_loading',
                'message': 'ML models are still loading. Please wait...'
            })
            
        return jsonify({
            'available_models': list(api.models.keys()),
            'total_models': len(api.models),
            'hand_detection': 'enhanced_two_hand_support',
            'status': 'ready'
        })
    except Exception as e:
        logger.error(f"Models endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    try:
        logger.info("Translate endpoint called")
        
        if not api:
            return jsonify({
                'success': False, 
                'error': 'ML models are still loading. Please wait and try again.',
                'status': 'loading'
            }), 503  # Service Unavailable
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        if not api.models:
            error_msg = 'No models loaded. Please check model files.'
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Process image and make prediction
        image_data = data['image']
        language_type = data.get('language_type', 'bisindo').lower()
        mirror_mode = data.get('mirror_mode', None)
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return jsonify({'success': False, 'error': f'Image processing failed: {e}'}), 400
        
        # Make prediction (implement this method based on your original code)
        prediction, confidence, message = "demo_result", 0.95, "Demo mode - models loading"
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'language_type': language_type,
            'dataset': language_type.upper(),
            'mirror_mode': mirror_mode,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize ML models in background after Flask starts
def start_background_loading():
    """Start ML model loading in background"""
    import threading
    thread = threading.Thread(target=init_ml_models)
    thread.daemon = True
    thread.start()
    logger.info("Started background ML model loading")

if __name__ == '__main__':
    print("\nENHANCED SIGN LANGUAGE API FOR RAILWAY")
    print("=" * 50)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
    print("✓ Basic Flask app ready for healthcheck")
    print("✓ ML models will load in background")
    print("=" * 50)
    
    # Start background loading
    start_background_loading()
    
    try:
        # Railway optimized server start
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")
        # Fallback for development
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
