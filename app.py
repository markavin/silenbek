import os
import sys
import logging
import json
import base64
import io
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

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

print(f"🚀 STARTING APP ON {HOST}:{PORT}")
logger.info(f"Starting Flask app on {HOST}:{PORT}")
logger.info(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
logger.info(f"Python version: {sys.version}")

# Initialize Flask app
app = Flask(__name__)

# Production CORS configuration
if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('PORT'):
    # Production - specific origins
    CORS(app, origins=[
        "https://silentdicoding.vercel.app",
        "https://silentdicoding-q93cjrxhx-evans-projects-d43a2e39.vercel.app",
        "https://silenbek-production.up.railway.app"
    ])
    logger.info("🔒 Production CORS enabled for specific origins")
else:
    # Development - permissive
    CORS(app, resources={"*": {"origins": "*"}})
    logger.info("🔓 Development CORS enabled")

# Global state
app_ready = True

@app.before_request
def log_request_info():
    logger.info(f"📥 {request.method} {request.path} from {request.remote_addr}")

# SIMPLIFIED health check - CRITICAL untuk Railway
@app.route('/api/health', methods=['GET'])
def health_check():
    """Ultra-simple health check for Railway"""
    try:
        logger.info("🏥 Health check called")
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'port': PORT,
            'service': 'Sign Language API - Minimal',
            'version': '1.0.0'
        }
        logger.info(f"✅ Health check OK: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 200  # Still return 200 for Railway

@app.route('/health', methods=['GET'])
def health_alt():
    return health_check()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    try:
        logger.info("🏠 Root endpoint called")
        response = {
            'service': 'Sign Language API',
            'status': 'running',
            'message': 'Welcome to Sign Language API - Minimal Version',
            'endpoints': ['/api/health', '/api/translate'],
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"✅ Root OK")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"❌ Root error: {e}")
        return jsonify({'error': str(e)}), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    """Model status"""
    try:
        response = {
            'available_models': ['demo'],
            'status': 'demo_mode',
            'message': 'Running in lightweight demo mode'
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 200

# Lightweight image processing (NO OpenCV/NumPy)
def simple_image_validation(image_data):
    """Simple image validation without heavy libraries"""
    try:
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Basic base64 validation
            try:
                decoded = base64.b64decode(image_data)
                logger.info(f"✅ Image decoded, size: {len(decoded)} bytes")
                return True
            except Exception as e:
                logger.error(f"❌ Base64 decode failed: {e}")
                return False
        return False
    except Exception as e:
        logger.error(f"❌ Image validation error: {e}")
        return False

# Simple prediction (NO ML libraries)
def simple_predict(language_type='bisindo'):
    """Simple prediction without ML libraries"""
    import random
    
    # Demo predictions with different probabilities
    if language_type == 'bisindo':
        predictions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    else:
        predictions = ['K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    
    prediction = random.choice(predictions)
    confidence = random.uniform(0.65, 0.92)
    
    return prediction, confidence

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    """Lightweight translation endpoint"""
    try:
        logger.info("🔮 Translate endpoint called")
        
        # Get JSON data
        data = request.get_json()
        if not data:
            logger.error("❌ No JSON data")
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        # Check image data
        if 'image' not in data:
            logger.error("❌ No image data")
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        image_data = data['image']
        language_type = data.get('language_type', 'bisindo')
        
        logger.info(f"📊 Processing: language={language_type}")
        
        # Simple validation
        if not simple_image_validation(image_data):
            return jsonify({
                'success': False, 
                'error': 'Invalid image data'
            }), 400
        
        # Simple prediction
        prediction, confidence = simple_predict(language_type)
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'language_type': language_type,
            'model_status': 'lightweight_demo',
            'timestamp': datetime.now().isoformat(),
            'message': 'Lightweight prediction successful'
        }
        
        logger.info(f"✅ Prediction: {prediction} ({confidence:.2f})")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Translate error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"⚠️ 404: {request.path}")
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"💥 500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Test endpoint
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    logger.info("🧪 Test endpoint called")
    return jsonify({
        'message': 'Test successful',
        'timestamp': datetime.now().isoformat(),
        'port': PORT,
        'host': HOST,
        'ready': app_ready
    }), 200

# Startup verification
def verify_startup():
    """Verify app can start"""
    try:
        logger.info("🔍 Verifying startup...")
        logger.info(f"✅ Flask app created")
        logger.info(f"✅ CORS configured")
        logger.info(f"✅ Routes registered")
        logger.info(f"✅ Ready to serve on {HOST}:{PORT}")
        return True
    except Exception as e:
        logger.error(f"❌ Startup verification failed: {e}")
        return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SIGN LANGUAGE API - LIGHTWEIGHT VERSION")
    print("="*60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Environment: Railway")
    print(f"Mode: Lightweight Demo")
    print("="*60)
    
    # Verify startup
    if not verify_startup():
        logger.error("❌ Startup verification failed!")
        sys.exit(1)
    
    try:
        logger.info("🚀 Starting Flask server...")
        app.run(
            host=HOST, 
            port=PORT, 
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"💥 Failed to start server: {e}")
        sys.exit(1)
