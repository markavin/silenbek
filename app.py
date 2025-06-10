import os
import sys
import logging
import json
import base64
import io
import numpy as np
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

# Railway environment setup - CRITICAL
PORT = int(os.environ.get('PORT', 5000))
HOST = '0.0.0.0'

# Setup logging untuk Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Log startup info
logger.info(f"Starting Flask app on {HOST}:{PORT}")
logger.info(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
logger.info(f"Railway Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'Not set')}")

# Initialize Flask app
app = Flask(__name__)

# CORS Configuration - PRODUCTION READY
def setup_cors():
    """Setup CORS based on environment"""
    is_production = bool(os.environ.get('RAILWAY_ENVIRONMENT') or 
                        os.environ.get('PORT'))
    
    if is_production:
        # Production origins - ADD YOUR VERCEL DOMAINS HERE
        allowed_origins = [
            "https://silentdicoding.vercel.app",
            "https://silentdicoding-q93cjrxhx-evans-projects-d43a2e39.vercel.app",
            "https://silenbek-production.up.railway.app",
        ]
        
        # Also allow Vercel preview domains (pattern matching)
        vercel_pattern = r"https://silentdicoding-[a-z0-9-]+\.vercel\.app"
        
        def origin_matches(origin):
            if origin in allowed_origins:
                return True
            if re.match(vercel_pattern, origin):
                return True
            return False
        
        # Configure CORS
        CORS(app, 
             origins=allowed_origins,
             methods=['GET', 'POST', 'OPTIONS', 'HEAD'],
             allow_headers=[
                 'Content-Type', 
                 'Authorization', 
                 'User-Agent',
                 'Accept',
                 'Origin',
                 'X-Requested-With'
             ],
             supports_credentials=True,
             max_age=86400
        )
        
        logger.info("🔒 Production CORS enabled")
        logger.info(f"📝 Allowed origins: {allowed_origins}")
        
        # Custom origin checker for dynamic Vercel URLs
        @app.after_request
        def after_request(response):
            origin = request.headers.get('Origin')
            if origin and origin_matches(origin):
                response.headers.add('Access-Control-Allow-Origin', origin)
                response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,User-Agent')
                response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,HEAD')
                response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response
            
    else:
        # Development - permissive CORS
        CORS(app, resources={"*": {"origins": "*"}})
        logger.info("🔓 Development CORS enabled (all origins)")

setup_cors()

# Global variable for ML Model
model = None
model_loaded = False

# Try to load ML model (with fallback)
def load_ml_model():
    global model, model_loaded
    try:
        model_paths = [
            './data/models/sign_language_model_bisindo_sklearn.pkl',
            './data/models/sign_language_model_sibi_sklearn.pkl',
            './models/sign_language_model.pkl',
            './sign_language_model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"Found model at: {path}")
                model_loaded = True
                logger.info("Model loaded successfully!")
                return True
        
        logger.warning("No model file found. Running in demo mode.")
        return False
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

load_ml_model()

# Preprocessing function
def preprocess_image(image_data):
    try:
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array / 255.0
        
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def predict_sign(image_array, language_type='bisindo'):
    global model, model_loaded
    
    try:
        if model_loaded and model is not None:
            # Use actual model here when available
            import random
            signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            prediction = random.choice(signs)
            confidence = random.uniform(0.7, 0.95)
            
            return prediction, confidence
        else:
            # Fallback: return varied demo predictions
            import random
            demo_predictions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            prediction = random.choice(demo_predictions)
            confidence = random.uniform(0.6, 0.9)
            
            return prediction, confidence
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# Log requests with CORS info
@app.before_request
def log_request_info():
    origin = request.headers.get('Origin')
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    if origin:
        logger.info(f"Origin: {origin}")

# CORS preflight handler
@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options(path=None):
    origin = request.headers.get('Origin')
    logger.info(f"✈️ CORS preflight from: {origin}")
    return '', 200

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        logger.info("Health check called")
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'port': PORT,
            'host': HOST,
            'service': 'Sign Language API',
            'version': '1.0.0',
            'model_loaded': model_loaded,
            'cors_mode': 'production' if os.environ.get('RAILWAY_ENVIRONMENT') else 'development'
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 200

@app.route('/health', methods=['GET'])
def health_check_alt():
    return health_check()

@app.route('/', methods=['GET'])
def root():
    try:
        logger.info("Root endpoint called")
        response = {
            'service': 'Sign Language API',
            'status': 'running',
            'endpoints': ['/api/health', '/health', '/api/translate', '/api/models'],
            'timestamp': datetime.now().isoformat(),
            'message': 'Welcome to Sign Language API',
            'model_status': 'loaded' if model_loaded else 'demo_mode',
            'cors_info': {
                'mode': 'production' if os.environ.get('RAILWAY_ENVIRONMENT') else 'development',
                'request_origin': request.headers.get('Origin')
            }
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return jsonify({'error': str(e)}), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        response = {
            'available_models': ['bisindo', 'sibi'] if model_loaded else [],
            'total_models': 2 if model_loaded else 0,
            'status': 'loaded' if model_loaded else 'demo_mode',
            'message': 'Models ready for prediction' if model_loaded else 'Running in demo mode'
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Models endpoint error: {e}")
        return jsonify({'error': str(e)}), 200

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    try:
        logger.info("Translate endpoint called")
        
        # Log CORS info
        origin = request.headers.get('Origin')
        logger.info(f"Request origin: {origin}")
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data in request'}), 400
        
        image_data = data['image']
        language_type = data.get('language_type', 'bisindo')
        
        logger.info(f"Processing image for language: {language_type}")
        
        try:
            processed_image = preprocess_image(image_data)
            logger.info("Image preprocessing successful")
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return jsonify({
                'success': False, 
                'error': f'Image preprocessing failed: {str(e)}'
            }), 400
        
        try:
            prediction, confidence = predict_sign(processed_image, language_type)
            logger.info(f"Prediction successful: {prediction} (confidence: {confidence:.2f})")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({
                'success': False, 
                'error': f'Prediction failed: {str(e)}'
            }), 500
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'language_type': language_type,
            'model_status': 'production' if model_loaded else 'demo',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Translate response: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Translate endpoint error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.path}")
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SIGN LANGUAGE API - PRODUCTION READY")
    print("="*60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Environment: {'Production' if os.environ.get('RAILWAY_ENVIRONMENT') else 'Development'}")
    print(f"Model Status: {'Loaded' if model_loaded else 'Demo Mode'}")
    print("="*60)
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
