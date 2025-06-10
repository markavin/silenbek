#!/usr/bin/env python3
# app.py - Fixed for Railway deployment

import os
import sys
import cv2
import numpy as np
import base64
import logging
import json
import mediapipe as mp
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import io
from PIL import Image
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Railway environment setup
PORT = int(os.environ.get('PORT', 5000))
HOST = '0.0.0.0'

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.data_preprocessing.feature_extractor import extract_features
    extract_features_available = True
    logger.info("Feature extractor imported successfully")
except ImportError as e:
    extract_features_available = False
    logger.warning(f"Feature extractor not available: {e}")

app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})

class EnhancedSignLanguageAPI:
    def __init__(self):
        self.models = {}
        self.project_root = project_root
        
        # Enhanced MediaPipe setup for better two-hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,  # Ensure 2 hands
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.4    # Lower threshold for tracking
        )
        
        self.load_models()
        
    def load_models(self):
        """Load available models"""
        logger.info("Loading models...")
        
        model_configs = [
            {
                'name': 'SIBI',
                'tensorflow_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi_tensorflow.h5'),
                'tensorflow_meta_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi_tensorflow_meta.pkl'),
                'sklearn_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi_sklearn.pkl'),
                'combined_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi.pkl'),
            },
            {
                'name': 'BISINDO',
                'tensorflow_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo_tensorflow.h5'),
                'tensorflow_meta_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo_tensorflow_meta.pkl'),
                'sklearn_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo_sklearn.pkl'),
                'combined_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo.pkl'),
            }
        ]
        
        for config in model_configs:
            model_info = {'available_models': []}
            
            # Load sklearn model (prefer this for stability)
            if os.path.exists(config['sklearn_path']):
                try:
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
    
    def preprocess_image_for_detection(self, image):
        """Enhanced preprocessing for better hand detection"""
        try:
            height, width = image.shape[:2]
            
            # Resize to optimal size
            target_size = 640
            if width > height:
                new_width = target_size
                new_height = int(height * target_size / width)
            else:
                new_height = target_size
                new_width = int(width * target_size / height)
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Enhanced preprocessing for better hand detection
            # 1. Improve contrast
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            enhanced = cv2.merge((l_channel, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Noise reduction while preserving hand edges
            denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def extract_landmarks_from_frame(self, image, mirror_mode=None):
        """Enhanced landmark extraction with better two-hand support"""
        try:
            processed_image = self.preprocess_image_for_detection(image)
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                logger.debug("No hands detected")
                return None
            
            landmarks_flat = []
            hand_data = []
            
            # Process all detected hands
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                wrist_x = hand_landmarks.landmark[0].x
                
                # Adjust handedness based on mirror mode
                if mirror_mode is True:
                    adjusted_label = hand_label
                elif mirror_mode is False:
                    adjusted_label = "Right" if hand_label == "Left" else "Left"
                else:
                    adjusted_label = hand_label
                
                hand_data.append({
                    'landmarks': hand_landmarks,
                    'original_label': hand_label,
                    'adjusted_label': adjusted_label,
                    'score': hand_score,
                    'wrist_x': wrist_x,
                    'index': i
                })
            
            num_hands = len(hand_data)
            # Fixed the f-string syntax error by creating the hand info separately
            hand_info = [(h['adjusted_label'], f"{h['score']:.2f}") for h in hand_data]
            logger.info(f"Detected {num_hands} hands: {hand_info}")
            
            # Enhanced hand sorting for consistency
            # For BISINDO (two-hand): sort by position (left to right)
            # For SIBI (one-hand): sort by confidence
            if num_hands >= 2:
                # Two-hand mode: sort by x position (left hand first)
                hand_data.sort(key=lambda x: x['wrist_x'])
            else:
                # Single-hand mode: sort by confidence
                hand_data.sort(key=lambda x: -x['score'])
            
            # Extract landmarks for up to 2 hands
            for hand_idx in range(2):
                if hand_idx < len(hand_data):
                    hand_landmarks = hand_data[hand_idx]['landmarks']
                    hand_label = hand_data[hand_idx]['adjusted_label']
                    hand_confidence = hand_data[hand_idx]['score']
                    
                    logger.debug(f"Processing hand {hand_idx}: {hand_label} (conf: {hand_confidence:.2f})")
                    
                    for landmark in hand_landmarks.landmark:
                        landmarks_flat.extend([
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z)
                        ])
                else:
                    # Pad with zeros for missing hand
                    landmarks_flat.extend([0.0] * 63)
            
            # Ensure exactly 126 values
            if len(landmarks_flat) != 126:
                landmarks_flat = landmarks_flat[:126]
                while len(landmarks_flat) < 126:
                    landmarks_flat.append(0.0)
            
            logger.info(f"Extracted landmarks: {len(landmarks_flat)} values for {num_hands} hands")
            return landmarks_flat
            
        except Exception as e:
            logger.error(f"Landmark extraction error: {e}")
            return None
    
    def extract_features_using_pipeline(self, landmarks_data, language):
        """Extract features using the same pipeline as training"""
        try:
            if extract_features_available:
                landmark_cols = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']]
                df_landmarks = pd.DataFrame([landmarks_data], columns=landmark_cols)
                
                features_df = extract_features(df_landmarks, perform_selection=False)
                
                if not features_df.empty:
                    features_for_prediction = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
                    return features_for_prediction
            
            return self.create_fallback_features(landmarks_data, language)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return self.create_fallback_features(landmarks_data, language)
    
    def create_fallback_features(self, landmarks_data, language):
        """Create fallback features"""
        try:
            features = {}
            
            hand1_data = landmarks_data[:63]
            hand2_data = landmarks_data[63:]
            
            feature_idx = 0
            
            for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
                if len(hand_data) >= 63:
                    x_coords = hand_data[::3]
                    y_coords = hand_data[1::3]
                    z_coords = hand_data[2::3]
                    
                    hand_exists = not all(x == 0 and y == 0 for x, y in zip(x_coords, y_coords))
                    
                    if hand_exists:
                        # Basic statistics
                        stats = [
                            np.mean(x_coords), np.std(x_coords), np.min(x_coords), np.max(x_coords),
                            np.mean(y_coords), np.std(y_coords), np.min(y_coords), np.max(y_coords),
                            np.mean(z_coords), np.std(z_coords), np.min(z_coords), np.max(z_coords),
                        ]
                        
                        for stat in stats:
                            if feature_idx < 80:
                                features[f'feature_{feature_idx}'] = float(stat) if np.isfinite(stat) else 0.0
                                feature_idx += 1
                        
                        # Distances from wrist
                        wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                        for tip_idx in [4, 8, 12, 16, 20]:
                            if tip_idx < len(x_coords) and feature_idx < 80:
                                dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                             (y_coords[tip_idx] - wrist_y)**2 + 
                                             (z_coords[tip_idx] - wrist_z)**2)
                                features[f'feature_{feature_idx}'] = float(dist) if np.isfinite(dist) else 0.0
                                feature_idx += 1
                        
                        # Hand geometry
                        if feature_idx < 80:
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            features[f'feature_{feature_idx}'] = float(width) if np.isfinite(width) else 0.0
                            feature_idx += 1
                            if feature_idx < 80:
                                features[f'feature_{feature_idx}'] = float(height) if np.isfinite(height) else 0.0
                                feature_idx += 1
                    else:
                        # Fill with zeros for missing hand
                        for i in range(17):  # 12 stats + 5 distances
                            if feature_idx < 80:
                                features[f'feature_{feature_idx}'] = 0.0
                                feature_idx += 1
            
            # Two-hand interaction features for BISINDO
            if len(hand1_data) >= 63 and len(hand2_data) >= 63:
                hand1_exists = not all(x == 0 and y == 0 for x, y in zip(hand1_data[::3], hand1_data[1::3]))
                hand2_exists = not all(x == 0 and y == 0 for x, y in zip(hand2_data[::3], hand2_data[1::3]))
                
                if hand1_exists and hand2_exists and feature_idx < 80:
                    # Distance between wrists
                    wrist1_x, wrist1_y = hand1_data[0], hand1_data[1]
                    wrist2_x, wrist2_y = hand2_data[0], hand2_data[1]
                    inter_wrist_dist = np.sqrt((wrist1_x - wrist2_x)**2 + (wrist1_y - wrist2_y)**2)
                    features[f'feature_{feature_idx}'] = float(inter_wrist_dist) if np.isfinite(inter_wrist_dist) else 0.0
                    feature_idx += 1
                    
                    # Relative positions
                    if feature_idx < 80:
                        features[f'feature_{feature_idx}'] = float(wrist1_x - wrist2_x) if np.isfinite(wrist1_x - wrist2_x) else 0.0
                        feature_idx += 1
                    if feature_idx < 80:
                        features[f'feature_{feature_idx}'] = float(wrist1_y - wrist2_y) if np.isfinite(wrist1_y - wrist2_y) else 0.0
                        feature_idx += 1
            
            # Pad to minimum features
            while feature_idx < 50:
                features[f'feature_{feature_idx}'] = 0.0
                feature_idx += 1
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Fallback feature creation failed: {e}")
            basic_features = {f'feature_{i}': 0.0 for i in range(50)}
            return pd.DataFrame([basic_features])
    
    def predict_with_model(self, features_df, language):
        """Make prediction using available model"""
        try:
            model_info = self.models[language]
            
            # Prefer sklearn model for stability
            if 'sklearn' in model_info['available_models']:
                return self.predict_sklearn(features_df, model_info['sklearn_model'])
            elif 'tensorflow' in model_info['available_models']:
                return self.predict_tensorflow(features_df, model_info['tensorflow_model'], model_info['tensorflow_meta'])
            else:
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Prediction failed for {language}: {e}")
            return None, 0.0
    
    def predict_sklearn(self, features_df, model_data):
        """Predict using sklearn model"""
        try:
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_names = model_data.get('feature_names')
            
            # Handle feature compatibility
            if feature_names:
                missing_features = set(feature_names) - set(features_df.columns)
                for feature in missing_features:
                    features_df[feature] = 0.0
                features_df = features_df[feature_names]
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Get confidence
            try:
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            except:
                confidence = 0.7
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Sklearn prediction error: {e}")
            return None, 0.0
    
    def predict_tensorflow(self, features_df, model, meta):
        """Predict using TensorFlow model"""
        try:
            scaler = meta.get('scaler')
            label_encoder = meta.get('label_encoder')
            feature_names = meta.get('feature_names')
            
            # Handle feature compatibility
            if feature_names:
                missing_features = set(feature_names) - set(features_df.columns)
                for feature in missing_features:
                    features_df[feature] = 0.0
                features_df = features_df[feature_names]
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            predictions_prob = model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(predictions_prob, axis=1)[0]
            confidence = np.max(predictions_prob)
            
            # Convert to label
            if label_encoder:
                prediction = label_encoder.inverse_transform([predicted_class_idx])[0]
            else:
                prediction = f"Class_{predicted_class_idx}"
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"TensorFlow prediction error: {e}")
            return None, 0.0
    
    def predict_sign(self, image, language_type='bisindo', mirror_mode=None):
        """Main prediction function with enhanced two-hand support"""
        try:
            language_type = language_type.upper()
            logger.info(f"Predicting for {language_type} (mirror_mode={mirror_mode})")
            
            if language_type not in self.models:
                available = list(self.models.keys())
                error_msg = f"Model not available for {language_type}. Available: {available}"
                logger.error(error_msg)
                return None, 0.0, error_msg
            
            # Extract landmarks with enhanced two-hand detection
            landmarks = self.extract_landmarks_from_frame(image, mirror_mode=mirror_mode)
            if landmarks is None:
                logger.warning("No hand landmarks detected")
                return "No hand detected", 0.0, "No hand landmarks detected"
            
            # Extract features
            features_df = self.extract_features_using_pipeline(landmarks, language_type)
            if features_df.empty:
                return None, 0.0, "Feature extraction failed"
            
            # Make prediction
            prediction, confidence = self.predict_with_model(features_df, language_type)
            
            if prediction is None:
                return None, 0.0, "Model prediction failed"
            
            # Check for two-hand detection (especially for BISINDO)
            hand1_exists = not all(landmarks[i] == 0 for i in range(0, 63, 3))
            hand2_exists = not all(landmarks[i] == 0 for i in range(63, 126, 3))
            hands_detected = sum([hand1_exists, hand2_exists])
            
            mirror_info = f" (mirror={mirror_mode})" if mirror_mode is not None else ""
            logger.info(f"SUCCESS{mirror_info}: {prediction} (confidence: {confidence:.3f}, hands: {hands_detected})")
            
            return prediction, confidence, f"Success - {hands_detected} hand(s) detected"
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0, f"Prediction error: {str(e)}"

# Initialize API
api = EnhancedSignLanguageAPI()

# RAILWAY HEALTH CHECK ENDPOINT (Simple and robust)
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check for Railway"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': list(api.models.keys()),
            'total_models': len(api.models),
            'mediapipe_ready': api.mp_hands is not None,
            'port': PORT,
            'host': HOST
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
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

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    try:
        logger.info("Enhanced translate endpoint called")
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        if not api.models:
            error_msg = 'No models loaded. Please train models first.'
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg}), 500
        
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
        
        # Make prediction with enhanced two-hand support
        prediction, confidence, message = api.predict_sign(image, language_type, mirror_mode=mirror_mode)
        
        response = {
            'success': prediction is not None,
            'prediction': prediction,
            'confidence': float(confidence) if confidence else 0.0,
            'language_type': language_type,
            'dataset': language_type.upper(),
            'mirror_mode': mirror_mode,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'available_models': list(api.models.keys()),
        'total_models': len(api.models),
        'hand_detection': 'enhanced_two_hand_support'
    })

if __name__ == '__main__':
    print("\nENHANCED TWO-HAND SIGN LANGUAGE API")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Models loaded: {list(api.models.keys())}")
    print(f"Total models: {len(api.models)}")
    print(f"MediaPipe ready: {api.mp_hands is not None}")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print("ENHANCEMENTS:")
    print("  ✓ Enhanced two-hand detection for BISINDO")
    print("  ✓ Better preprocessing and CLAHE")
    print("  ✓ Mirror-aware handedness processing")
    print("  ✓ Improved hand sorting algorithms")
    print("  ✓ Sklearn + TensorFlow model support")
    print("  ✓ Railway optimized health checks")
    
    if not api.models:
        print("\nNO MODELS LOADED!")
        print("Run: python main.py")
    else:
        print("\nAPI READY FOR TWO-HAND PREDICTIONS!")
        for lang, info in api.models.items():
            available_types = ', '.join(info['available_models'])
            print(f"   {lang}: {available_types}")
    
    print("=" * 50)
    
    try:
        # Railway optimized server start
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")
        # Fallback untuk development
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
