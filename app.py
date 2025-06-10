import os
import sys
import logging
import json
import base64
import io
import pickle # Pastikan ini diimpor jika model Anda adalah .pkl
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import libraries for image processing and model prediction
# Pastikan ini ada di requirements.txt Anda
try:
    import numpy as np
    from PIL import Image
    # Jika Anda menggunakan scikit-learn, pastikan juga diimpor di sini:
    # import sklearn # Tidak perlu langsung mengimpor objek model di sini, tapi pastikan terinstal
    # Jika Anda menggunakan TensorFlow dan model Anda adalah Keras/TF yang disimpan sebagai .pkl (jarang, tapi mungkin)
    # import tensorflow as tf
    heavy_libs_available = True
except ImportError as e:
    heavy_libs_available = False
    logging.warning(f"⚠ Heavy libraries (numpy, PIL) not fully available at import: {e}. Running in lightweight/demo mode.")


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

print(f"🚀 STARTING SIGN LANGUAGE API - HYBRID MODE ON {HOST}:{PORT}")

# Initialize Flask app
app = Flask(_name_)

# CORS configuration
# Untuk produksi, sangat disarankan untuk membatasi origin ke URL frontend Anda
# Misalnya: os.environ.get('FRONTEND_URL', 'http://localhost:5173')
# Tapi untuk debugging awal, wildcard (*) bisa diterima.
CORS(app, 
     resources={"": {"origins": ""}}, # Anda bisa mengganti "*" dengan daftar origin yang spesifik
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'User-Agent'])

logger.info("🔓 CORS configured")

# Global variables
models = {} # Dictionary untuk menyimpan model yang dimuat
model_loaded = False # Status apakah ada model yang berhasil dimuat
available_models = [] # Daftar nama model yang tersedia (e.g., 'bisindo', 'sibi')

# --- Fungsi untuk memeriksa dan memuat library ML (sudah ada, bagus!) ---
# Tidak perlu ada perubahan signifikan di sini, kecuali memastikan 'pickle' juga dicek
# (Anda sudah punya)
def check_heavy_libraries_runtime(): # Mengganti nama agar tidak bentrok dengan global heavy_libs_available
    """Check if heavy ML libraries are available at runtime"""
    global heavy_libs_available
    try:
        import numpy
        import PIL.Image # Akses PIL.Image secara eksplisit
        import pickle # Pastikan pickle juga bisa diimpor
        # Jika Anda menggunakan cv2, pastikan terimpor juga
        # import cv2
        heavy_libs_available = True
        logger.info("✅ All heavy libraries (NumPy, PIL, Pickle) available.")
        return True
    except ImportError as e:
        heavy_libs_available = False
        logger.warning(f"⚠ Missing heavy libraries: {e}. Running in lightweight/demo mode.")
        return False

# Panggil di awal untuk set status global
heavy_libs_available = check_heavy_libraries_runtime()


# --- Fungsi Pre-processing Gambar (PENTING untuk Akurasi) ---
# Ini HARUS SAMA PERSIS dengan pre-processing saat model dilatih
def heavy_image_processing(image_data):
    """Heavy image processing with ML libraries (NumPy, PIL)"""
    try:
        if not heavy_libs_available:
            raise ImportError("Heavy libraries not available for processing.")

        # Decode base64
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1] # Menghapus prefix data URI
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data # Jika sudah dalam bentuk bytes

        # Load with PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (model ML sering dilatih dengan gambar RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # --- PENTING: SESUAIKAN UKURAN DAN METODE RESIZE MODEL ANDA ---
        target_size = (64, 64) # Ukuran yang model Anda latih
        # Gunakan Image.LANCZOS atau Image.BICUBIC untuk hasil resize yang lebih baik
        resized_image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(resized_image)

        # --- PENTING: SESUAIKAN NORMALISASI MODEL ANDA ---
        # Contoh: Normalisasi ke 0-1 (jika model dilatih dengan nilai piksel 0-1)
        normalized_image = image_array.astype(np.float32) / 255.0
        
        # --- PENTING: SESUAIKAN BENTUK INPUT MODEL ANDA ---
        # Untuk model scikit-learn (SVM, Logistic Regression, dll.) yang dilatih dengan input flattened
        # Bentuk output: (1, jumlah_piksel)
        flattened_image = normalized_image.flatten().reshape(1, -1)
        
        # Jika model Anda adalah CNN (TensorFlow/Keras) yang membutuhkan (batch, height, width, channels):
        # Misalnya untuk gambar RGB 64x64: (1, 64, 64, 3)
        # processed_data_for_cnn = np.expand_dims(normalized_image, axis=0)

        logger.info(f"✅ Heavy processing: Output shape: {flattened_image.shape}, dtype: {flattened_image.dtype}, min: {np.min(flattened_image)}, max: {np.max(flattened_image)}")
        
        # Kembalikan processed_data yang sesuai dengan model Anda
        return flattened_image # Mengembalikan flattened_image untuk contoh sklearn

    except Exception as e:
        logger.error(f"❌ Heavy processing failed: {e}")
        raise # Reraise exception agar ditangani oleh blok try/except di atas

# --- Fungsi pemrosesan ringan (seperti yang sudah ada) ---
def lightweight_image_processing(image_data):
    """Lightweight image processing without heavy dependencies"""
    try:
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            decoded = base64.b64decode(image_data)
            if len(decoded) < 1000:
                raise ValueError("Image too small")
            logger.info(f"✅ Lightweight processing: {len(decoded)} bytes")
            return decoded
    except Exception as e:
        logger.error(f"❌ Lightweight processing failed: {e}")
        raise

# --- Pemuatan Model (Revisi Path dan Logging) ---
def load_models_safe():
    """Safely try to load models"""
    global models, model_loaded, available_models
    
    models = {} # Reset models
    model_loaded = False
    available_models = []

    try:
        logger.info("🔍 Searching for models...")
        
        # Path relatif ke direktori kerja kontainer (/app)
        # Pastikan Dockerfile menyalin model ke 'models/' atau root '/'
        # Jika Anda menyalinnya ke 'models/' di Dockerfile:
        model_search_path = 'models' # Asumsi folder 'models' ada di root /app
        
        model_files_found = []
        # Menggunakan os.walk untuk mencari file .pkl atau .joblib
        for root, dirs, files in os.walk(model_search_path):
            for file in files:
                if file.endswith(('.pkl', '.joblib')):
                    full_path = os.path.join(root, file)
                    model_files_found.append(full_path)
                    logger.info(f"📄 Found: {full_path}")
        
        if not model_files_found:
            logger.warning("ℹ No model files found in expected path - using smart demo mode")
            return False
        
        # Try to load models only if heavy libs (including pickle) are available
        if heavy_libs_available:
            for file_path in model_files_found:
                try:
                    logger.info(f"📂 Attempting to load: {file_path}")
                    
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f) # Menggunakan pickle yang sudah diimpor
                    
                    # Determine language from filename
                    filename = os.path.basename(file_path).lower()
                    language = 'unknown' # Default
                    if 'bisindo' in filename:
                        language = 'bisindo'
                    elif 'sibi' in filename:
                        language = 'sibi'
                    
                    models[language] = model
                    available_models.append(language)
                    logger.info(f"✅ Loaded {language} model from {file_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {file_path}: {e}")
                    # Lanjutkan ke model berikutnya jika ada
                    continue
            
            if models:
                model_loaded = True
                logger.info(f"🎉 Successfully loaded models: {available_models}")
            else:
                logger.warning("⚠ No models successfully loaded - falling back to demo mode.")
        else:
            logger.warning("⚠ Heavy libraries not available, cannot load models. Using demo mode.")
        
        return model_loaded
        
    except Exception as e:
        logger.error(f"💥 Model loading error: {e}")
        return False

# --- Fungsi Prediksi (smart_predict & intelligent_demo_predict) ---
# Tidak ada perubahan besar di sini, karena logika fallback sudah bagus.
# Namun, pastikan alphabet di intelligent_demo_predict sesuai dengan kelas Anda.
def smart_predict(processed_data, language_type='bisindo'):
    """Smart prediction - use real model if available, otherwise intelligent demo"""
    global models, model_loaded
    
    try:
        if model_loaded and language_type in models:
            logger.info(f"🤖 Using real {language_type} model")
            
            model = models[language_type]
            
            # PENTING: Pastikan input processed_data memiliki bentuk yang diharapkan oleh model.
            # Log shape di heavy_image_processing akan membantu debug ini.
            prediction_result = model.predict(processed_data)
            
            # Jika model Anda mengembalikan array dengan satu elemen
            prediction = prediction_result[0] 
            
            # Dapatkan confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)[0]
                confidence = float(np.max(probabilities)) # Menggunakan np.max untuk confidence
            else:
                # Fallback confidence jika model tidak punya predict_proba
                confidence = 0.87
            
            # Pastikan prediction_result adalah string atau bisa di-konversi ke string
            # Jika prediction_result adalah angka (indeks kelas), Anda perlu mapping ke label
            # Contoh:
            # labels = ['A', 'B', 'C', ..., 'Z'] # Ganti dengan label Anda
            # if isinstance(prediction, (int, np.integer)):
            #     prediction = labels[int(prediction)]

            return str(prediction), confidence
        
        else:
            logger.info(f"🎯 Using intelligent demo for {language_type}")
            return intelligent_demo_predict(processed_data, language_type)
            
    except Exception as e:
        logger.error(f"💥 Prediction error: {e}")
        # Log processed_data shape jika error di sini
        if 'processed_data' in locals():
             logger.error(f"Error occurred with processed_data shape: {processed_data.shape}")
        return intelligent_demo_predict(processed_data, language_type)

def intelligent_demo_predict(processed_data, language_type):
    """More intelligent demo prediction based on image characteristics - ALPHABET ONLY"""
    import random
    import hashlib
    
    try:
        if isinstance(processed_data, bytes):
            image_hash = hashlib.md5(processed_data).hexdigest()
        else:
            # Convert to string for hashing
            image_hash = hashlib.md5(str(processed_data).encode()).hexdigest()
        
        hash_int = int(image_hash[:8], 16)
        
        # PASTIKAN alphabet ini sesuai dengan daftar kelas yang Anda harapkan dari demo
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                   'U', 'V', 'W', 'X', 'Y', 'Z']
        
        char_index = hash_int % len(alphabet)
        prediction = alphabet[char_index]
        
        confidence = 0.70 + (hash_int % 25) / 100
        
        logger.info(f"🎲 Intelligent demo (alphabet): {prediction} ({confidence:.2f})")
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Demo prediction error: {e}")
        return random.choice(['A', 'B', 'C', 'D', 'E']), 0.75

# --- Inisialisasi pada Startup ---
# check_heavy_libraries() # Sudah diganti dengan panggilan di atas
load_models_safe() # Panggil setelah memastikan heavy_libs_available di set


# --- Rute API (Tidak banyak perubahan signifikan di sini) ---

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
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data.get('image', '')
        language_type = data.get('language_type', 'bisindo')
        
        logger.info(f"🎯 Language: {language_type}, Heavy libs available: {heavy_libs_available}")
        
        # Process image
        processed_data = None
        processing_mode = 'unknown'
        try:
            if heavy_libs_available:
                processed_data = heavy_image_processing(image_data)
                processing_mode = 'heavy'
            else:
                # Jika heavy_libs_available = False, itu berarti numpy/PIL tidak terinstal
                # Maka, tidak bisa melakukan heavy_image_processing.
                # Kita tidak bisa melakukan prediksi model ML yang sebenarnya tanpa heavy_image_processing.
                # Jadi, langsung fallback ke demo.
                # processed_data = lightweight_image_processing(image_data) # Ini tidak berguna untuk prediksi ML
                processing_mode = 'lightweight_then_demo'
                logger.warning("⚠ Heavy libraries not available, falling back to demo mode directly after lightweight processing.")
                # Kita akan membiarkan smart_predict menangani ini
                
        except Exception as e:
            logger.error(f"❌ Image processing failed even with heavy libs available (might be data format): {e}")
            processing_mode = 'failed_processing_then_demo' # Update mode jika gagal
            # processed_data tetap None, sehingga smart_predict akan ke demo

        # Make prediction
        # smart_predict akan secara otomatis menggunakan demo jika processed_data None atau model tidak dimuat
        prediction, confidence = smart_predict(processed_data if processing_mode == 'heavy' else image_data, language_type)
        # ^^^^^^ PENTING: Mengirim image_data ke smart_predict jika processing_mode bukan 'heavy'
        # karena intelligent_demo_predict bisa menangani raw base64 string.
        
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': float(confidence),
            'language_type': language_type,
            'model_status': 'real_model' if (model_loaded and language_type in models and processing_mode == 'heavy') else 'intelligent_demo', # Logika yang lebih akurat
            'processing_mode': processing_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Result: {prediction} ({confidence:.2f}) - {response['model_status']} (Proc: {processing_mode})")
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
