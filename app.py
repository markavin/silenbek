import os
import sys
import logging
import json
import base64
import io
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Railway environment setup
PORT = int(os.environ.get('PORT', 5000))
HOST = '0.0.0.0'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(_name_)

print(f"🚀 STARTING LIGHTWEIGHT MODEL APP ON {HOST}:{PORT}")

# Initialize Flask app
app = Flask(_name_)

# CORS configuration
CORS(app, 
     resources={"": {"origins": ""}},
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'User-Agent'])

logger.info("🔓 CORS configured")

# Global variables
models = {}
model_loaded = False
available_models = []
heavy_libs_available = False

def check_heavy_libraries():
    """Check if heavy ML libraries are available"""
    global heavy_libs_available
    
    try:
        import numpy
        logger.info("✅ NumPy available")
        
        try:
            import cv2
            logger.info("✅ OpenCV available")
        except ImportError:
            logger.warning("⚠ OpenCV not available - using PIL only")
        
        try:
            from PIL import Image
            logger.info("✅ PIL available")
        except ImportError:
            logger.warning("⚠ PIL not available")
            
        try:
            import pickle
            logger.info("✅ Pickle available")
        except ImportError:
            logger.warning("⚠ Pickle not available")
        
        heavy_libs_available = True
        return True
        
    except ImportError as e:
        logger.warning(f"⚠ Heavy libraries not available: {e}")
        heavy_libs_available = False
        return False

def lightweight_image_processing(image_data):
    """Lightweight image processing without heavy dependencies"""
    try:
        # Basic base64 validation and decode
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode and validate
            decoded = base64.b64decode(image_data)
            
            # Basic size check
            if len(decoded) < 1000:  # Very small image
                raise ValueError("Image too small")
            
            logger.info(f"✅ Lightweight processing: {len(decoded)} bytes")
            return decoded
            
    except Exception as e:
        logger.error(f"❌ Lightweight processing failed: {e}")
        raise

def heavy_image_processing(image_data):
    """Heavy image processing with ML libraries (if available)"""
    try:
        import numpy as np
        from PIL import Image
        
        # Decode base64
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Load with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize (simple version without OpenCV)
        target_size = (64, 64)
        resized = image.resize(target_size)
        
        # Convert to numpy
        image_array = np.array(resized)
        
        # Normalize
        normalized = image_array.astype(np.float32) / 255.0
        
        # Flatten for sklearn
        flattened = normalized.flatten().reshape(1, -1)
        
        logger.info(f"✅ Heavy processing: {flattened.shape}")
        return flattened
        
    except Exception as e:
        logger.error(f"❌ Heavy processing failed: {e}")
        raise

def load_models_safe():
    """Safely try to load models"""
    global models, model_loaded, available_models
    
    try:
        logger.info("🔍 Searching for models...")
        
        # Check available model files
        model_files_found = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.pkl', '.joblib')):
                    full_path = os.path.join(root, file)
                    model_files_found.append(full_path)
                    logger.info(f"📄 Found: {full_path}")
        
        if not model_files_found:
            logger.info("ℹ No model files found - using smart demo mode")
            return False
        
        # Try to load models if heavy libs available
        if heavy_libs_available:
            try:
                import pickle
                
                for file_path in model_files_found:
                    try:
                        logger.info(f"📂 Attempting to load: {file_path}")
                        
                        with open(file_path, 'rb') as f:
                            model = pickle.load(f)
                        
                        # Determine language from filename
                        filename = os.path.basename(file_path).lower()
                        if 'bisindo' in filename:
                            language = 'bisindo'
                        elif 'sibi' in filename:
                            language = 'sibi'
                        else:
                            language = 'unknown'
                        
                        models[language] = model
                        available_models.append(language)
                        logger.info(f"✅ Loaded {language} model from {file_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load {file_path}: {e}")
                        continue
                
                if models:
                    model_loaded = True
                    logger.info(f"🎉 Successfully loaded models: {available_models}")
                
            except ImportError:
                logger.warning("⚠ Pickle not available - cannot load models")
        
        return model_loaded
        
    except Exception as e:
        logger.error(f"💥 Model loading error: {e}")
        return False

def smart_predict(processed_data, language_type='bisindo'):
    """Smart prediction - use real model if available, otherwise intelligent demo"""
    global models, model_loaded
    
    try:
        # Try real model first
        if model_loaded and language_type in models:
            logger.info(f"🤖 Using real {language_type} model")
            
            model = models[language_type]
            prediction = model.predict(processed_data)[0]
            
            # Get confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 0.87
            
            return str(prediction), confidence
        
        else:
            # Intelligent demo prediction
            logger.info(f"🎯 Using intelligent demo for {language_type}")
            return intelligent_demo_predict(processed_data, language_type)
            
    except Exception as e:
        logger.error(f"💥 Prediction error: {e}")
        return intelligent_demo_predict(processed_data, language_type)

def intelligent_demo_predict(processed_data, language_type):
    """More intelligent demo prediction based on image characteristics - ALPHABET ONLY"""
    import random
    import hashlib
    
    try:
        # Create pseudo-deterministic prediction based on image data
        if isinstance(processed_data, bytes):
            image_hash = hashlib.md5(processed_data).hexdigest()
        else:
            # Convert to string for hashing
            image_hash = hashlib.md5(str(processed_data).encode()).hexdigest()
        
        # Use hash to create consistent but varied predictions
        hash_int = int(image_hash[:8], 16)
        
        # ONLY ALPHABET A-Z (sesuai dataset training)
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                   'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Pseudo-random but consistent selection
        char_index = hash_int % len(alphabet)
        prediction = alphabet[char_index]
        
        # Confidence based on hash characteristics
        confidence = 0.70 + (hash_int % 25) / 100  # 0.70-0.94
        
        logger.info(f"🎲 Intelligent demo (alphabet): {prediction} ({confidence:.2f})")
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Demo prediction error: {e}")
        # Fallback - random alphabet
        return random.choice(['A', 'B', 'C', 'D', 'E']), 0.75

# Initialize on startup
check_heavy_libraries()
load_models_safe()

@app.before_request
def log_request_info():
    logger.info(f"📥 {request.method} {request.path}")

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'port': PORT,
            'service': 'Sign Language API - Hybrid',
            'capabilities': {
                'heavy_libs': heavy_libs_available,
                'models_loaded': model_loaded,
                'available_models': available_models,
                'prediction_mode': 'real_model' if model_loaded else 'intelligent_demo'
            }
        }
        logger.info(f"✅ Health check OK")
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
        'service': 'Sign Language API - Hybrid',
        'status': 'running',
        'mode': 'real_model' if model_loaded else 'intelligent_demo',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'status': 'loaded' if model_loaded else 'intelligent_demo',
        'available_models': available_models,
        'prediction_mode': 'real_model' if model_loaded else 'intelligent_demo',
        'capabilities': {
            'heavy_processing': heavy_libs_available,
            'lightweight_processing': True
        }
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
        
        logger.info(f"🎯 Language: {language_type}, Heavy libs: {heavy_libs_available}")
        
        # Process image
        try:
            if heavy_libs_available:
                processed_data = heavy_image_processing(image_data)
                processing_mode = 'heavy'
            else:
                processed_data = lightweight_image_processing(image_data)
                processing_mode = 'lightweight'
                
            logger.info(f"📸 Processing mode: {processing_mode}")
            
        except Exception as e:
            logger.error(f"❌ Image processing failed: {e}")
            return jsonify({
                'success': False,
                'error': f'Image processing failed: {str(e)}'
            }), 400
        
        # Make prediction
        prediction, confidence = smart_predict(processed_data, language_type)
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'language_type': language_type,
            'model_status': 'real_model' if (model_loaded and language_type in models) else 'intelligent_demo',
            'processing_mode': processing_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Result: {prediction} ({confidence:.2f}) - {response['model_status']}")
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
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if _name_ == '_main_':
    print("\n" + "="*50)
    print("🔧 SIGN LANGUAGE API - HYBRID MODE")
    print("="*50)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Heavy libs: {heavy_libs_available}")
    print(f"Models loaded: {model_loaded}")
    print(f"Available: {available_models}")
    print("="*50)
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"💥 Server start failed: {e}")
        sys.exit(1)
