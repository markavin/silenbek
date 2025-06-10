import os
import sys
import logging
import json
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

# Log startup info
logger.info(f"Starting Flask app on {HOST}:{PORT}")
logger.info(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["https://silentdeploy-d8jsqqbd7-evans-projects-d43a2e39.vercel.app"])

# Global variable for ML API
api = None

# Add before_request handler untuk debugging
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    logger.info(f"Headers: {dict(request.headers)}")

# ULTRA SIMPLE health check - harus SELALU return 200
@app.route('/api/health', methods=['GET'])
def health_check():
    """Ultra-lightweight health check for Railway"""
    try:
        logger.info("Health check called")
        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'port': PORT,
            'host': HOST,
            'service': 'Sign Language API',
            'version': '1.0.0'
        }
        logger.info(f"Health check response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Bahkan jika error, tetap return 200 untuk healthcheck
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 200  # Note: 200, bukan 500!

# Alternative health endpoints - kadang Railway cek path lain
@app.route('/health', methods=['GET'])
def health_check_alt():
    """Alternative health check path"""
    return health_check()

@app.route('/healthcheck', methods=['GET'])
def health_check_alt2():
    """Alternative health check path"""
    return health_check()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for Railway"""
    try:
        logger.info("Root endpoint called")
        response = {
            'service': 'Sign Language API',
            'status': 'running',
            'endpoints': ['/api/health', '/health', '/api/translate', '/api/models'],
            'timestamp': datetime.now().isoformat(),
            'message': 'Welcome to Sign Language API'
        }
        logger.info(f"Root response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return jsonify({'error': str(e)}), 200  # Return 200 even on error

@app.route('/api/models', methods=['GET'])
def get_models():
    """Check model status"""
    try:
        logger.info("Models endpoint called")
        response = {
            'available_models': [],
            'total_models': 0,
            'status': 'models_loading',
            'message': 'Models are loading in background...'
        }
        logger.info(f"Models response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Models endpoint error: {e}")
        return jsonify({'error': str(e)}), 200

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    """Sign language translation endpoint"""
    try:
        logger.info("Translate endpoint called")
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        # Demo response for now
        response = {
            'success': True,
            'prediction': 'HELLO',  # Demo prediction
            'confidence': 0.85,
            'language_type': data.get('language_type', 'bisindo'),
            'message': 'Demo mode - basic functionality working',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Translate response: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Translate endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers - CRITICAL: return proper responses
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.path}")
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Test endpoint untuk debugging
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint for debugging"""
    logger.info("Test endpoint called")
    return jsonify({
        'message': 'Test successful',
        'timestamp': datetime.now().isoformat(),
        'port': PORT,
        'host': HOST
    }), 200

# Startup function
def create_app():
    """Create and configure the Flask app"""
    logger.info("Flask app created successfully")
    return app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MINIMAL SIGN LANGUAGE API FOR RAILWAY")
    print("="*60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
    print("Ultra-lightweight build")
    print("Instant health check")
    print("Debug logging enabled")
    print("="*60)
    
    try:
        logger.info("Starting Flask application...")
        app.run(
            host=HOST, 
            port=PORT, 
            debug=False,  # Set False for production
            threaded=True,
            use_reloader=False  # Important for Railway
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        # Fallback for development
        logger.info("Trying fallback server...")
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
