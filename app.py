#!/usr/bin/env python3
# app.py - FIXED VERSION - Identical preprocessing as camera_test.py

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
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests

# Setup logging
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
    logger.info(f"Added {project_root} to sys.path")

try:
    from src.data_preprocessing.feature_extractor import extract_features
    extract_features_available = True
    logger.info("Feature extractor imported successfully")
except ImportError as e:
    extract_features_available = False
    logger.error(f"Feature extractor not available: {e}")

app = Flask(__name__)

HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))

# CORS configuration
FRONTEND_URLS = [
    'https://silent-sign.vercel.app',
    'https://silent-signl-git-main-mark-alvins-projects-95223802.vercel.app',
    'https://silent-signl-82ztgrwmy-mark-alvins-projects-95223802.vercel.app',
    'http://localhost:3000',
    'http://localhost:5173',
    'http://127.0.0.1:3000',
    'https://localhost:3000'
]

CORS_ORIGINS = os.environ.get('FRONTEND_URL', ','.join(FRONTEND_URLS)).split(',')

CORS_RESOURCES = {
    r"/*": {
        "origins": CORS_ORIGINS,
        "methods": ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        "allow_headers": [
            'Content-Type', 
            'Authorization', 
            'User-Agent',
            'Accept',
            'Origin',
            'X-Requested-With'
        ],
        "supports_credentials": False,
        "max_age": 600
    }
}

CORS(app, resources=CORS_RESOURCES, methods=['GET', 'POST', 'OPTIONS'], 
     allow_headers=['Content-Type', 'Authorization', 'User-Agent', 'Accept', 'Origin', 'X-Requested-With'])

logger.info(f"CORS configured for domains: {CORS_ORIGINS}")

def download_model_files():
    """Download all model files including TensorFlow models"""
    model_urls = {
        "data/models/sign_language_model_bisindo_feature_names.pkl": "https://drive.google.com/uc?export=download&id=1W4Ad2TqAKcyGsb1E1CRhr6GhdowrAtNc",
        "data/models/sign_language_model_bisindo_sklearn.pkl": "https://drive.google.com/uc?export=download&id=1ItILhbuagMgQnG2q0fdM2qgDEA7EEkZn",
        "data/models/sign_language_model_bisindo_tensorflow_meta.pkl": "https://drive.google.com/uc?export=download&id=120qqIGU3pJMpvrh-nZQEjaffKbC8Ue8i",
        "data/models/sign_language_model_bisindo_tensorflow.h5": "https://drive.google.com/uc?export=download&id=1BlcRtHXGnp2G__aqJYgEnpZJ-4zaJVzf",
        "data/models/sign_language_model_bisindo.pkl": "https://drive.google.com/uc?export=download&id=1SHUZ_hAoh85Phcug6VynKHiro11JKFhy",
        "data/models/sign_language_model_sibi_feature_names.pkl": "https://drive.google.com/uc?export=download&id=1FVsG99ETKKGtUEYyTiLogpRhQVhKj7EB",
        "data/models/sign_language_model_sibi_sklearn.pkl": "https://drive.google.com/uc?export=download&id=1b51N3nFffWY51hPxAtMSOAtA6S_tKUX1",
        "data/models/sign_language_model_sibi_tensorflow_meta.pkl": "https://drive.google.com/uc?export=download&id=1yM25Vc6km_Rf0CsXBghpzqcgxo0G69mQ",
        "data/models/sign_language_model_sibi_tensorflow.h5": "https://drive.google.com/uc?export=download&id=1hvUYvyLNvCOueufW2IqL3BCxcLT1rk27",
        "data/models/sign_language_model_sibi.pkl": "https://drive.google.com/uc?export=download&id=1iB8LGBd875MfP6aScSPtERq9cH7ZR2pj"
    }

    for local_path, url in model_urls.items():
        if not os.path.exists(local_path):
            logger.info(f"Downloading model file to {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                response = requests.get(url, timeout=60)
                content_type = response.headers.get('Content-Type', '')
                if response.status_code == 200 and 'text/html' not in content_type.lower():
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Model saved: {local_path}")
                else:
                    logger.error(f"Invalid file content for {url}")
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")

download_model_files()

class FixedSignLanguageAPI:
    def __init__(self): 
        self.models = {}
        self.project_root = project_root
        
        # MediaPipe setup - IDENTICAL to camera_test.py
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.load_models()
        
    def load_models(self):
        """Load both sklearn and TensorFlow models with fixed compatibility"""
        logger.info("Loading models...")
        
        model_base_path = Path(self.project_root) / 'data' / 'models'
        
        model_configs = [
            {
                'name': 'SIBI',
                'sklearn_path': model_base_path / 'sign_language_model_sibi_sklearn.pkl',
                'tensorflow_path': model_base_path / 'sign_language_model_sibi_tensorflow.h5',
                'tensorflow_meta_path': model_base_path / 'sign_language_model_sibi_tensorflow_meta.pkl',
            },
            {
                'name': 'BISINDO',
                'sklearn_path': model_base_path / 'sign_language_model_bisindo_sklearn.pkl',
                'tensorflow_path': model_base_path / 'sign_language_model_bisindo_tensorflow.h5',
                'tensorflow_meta_path': model_base_path / 'sign_language_model_bisindo_tensorflow_meta.pkl',
            }
        ]
        
        for config in model_configs:
            model_info = {'available_models': []}
            
            # Load sklearn model
            if config['sklearn_path'].exists():
                try:
                    sklearn_data = joblib.load(config['sklearn_path'])
                    if isinstance(sklearn_data, dict) and 'model' in sklearn_data:
                        if self.validate_sklearn_model(sklearn_data, config['name']):
                            model_info['sklearn_model'] = sklearn_data
                            model_info['available_models'].append('sklearn')
                            logger.info(f"{config['name']}: Sklearn model loaded successfully")
                        else:
                            logger.warning(f"{config['name']}: Sklearn model validation failed")
                    else:
                        logger.warning(f"{config['name']}: Invalid sklearn model format")
                except Exception as e:
                    logger.warning(f"{config['name']}: Sklearn load failed - {e}")
            else:
                logger.info(f"{config['name']}: Sklearn model file not found")

            # Load TensorFlow model with compatibility fixes
            if config['tensorflow_path'].exists() and config['tensorflow_meta_path'].exists():
                try:
                    import tensorflow as tf
                    
                    # Load with custom options for compatibility
                    tf_model = tf.keras.models.load_model(
                        str(config['tensorflow_path']),
                        custom_objects=None,
                        compile=False
                    )
                    
                    # Recompile with current TensorFlow version
                    tf_model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    tf_meta = joblib.load(config['tensorflow_meta_path'])
                    
                    if self.validate_tensorflow_model(tf_model, tf_meta, config['name']):
                        model_info['tensorflow_model'] = tf_model
                        model_info['tensorflow_meta'] = tf_meta
                        model_info['available_models'].append('tensorflow')
                        logger.info(f"{config['name']}: TensorFlow model loaded successfully")
                    else:
                        logger.warning(f"{config['name']}: TensorFlow model validation failed")
                        
                except ImportError:
                    logger.warning(f"{config['name']}: TensorFlow not available")
                except Exception as e:
                    logger.warning(f"{config['name']}: TensorFlow load failed - {e}")
                    try:
                        import tensorflow.compat.v1 as tf_v1
                        tf_v1.disable_v2_behavior()
                        
                        tf_model = tf.keras.models.load_model(
                            str(config['tensorflow_path']),
                            compile=False
                        )
                        tf_meta = joblib.load(config['tensorflow_meta_path'])
                        
                        model_info['tensorflow_model'] = tf_model
                        model_info['tensorflow_meta'] = tf_meta
                        model_info['available_models'].append('tensorflow')
                        logger.info(f"{config['name']}: TensorFlow model loaded with v1 compatibility")
                        
                    except Exception as e2:
                        logger.error(f"{config['name']}: All TensorFlow loading methods failed - {e2}")
            else:
                logger.info(f"{config['name']}: TensorFlow model files not found")
            
            if model_info['available_models']:
                self.models[config['name']] = model_info
        
        logger.info(f"Loaded {len(self.models)} language models: {list(self.models.keys())}")
        for lang, info in self.models.items():
            logger.info(f"  {lang}: {', '.join(info['available_models'])}")
    
    def validate_tensorflow_model(self, model, meta, language):
        """Validate TensorFlow model with compatibility fixes"""
        try:
            input_shape = model.input_shape[1:]
            
            if len(input_shape) == 1:
                test_data = np.random.rand(1, input_shape[0]).astype(np.float32) * 0.1
            elif len(input_shape) == 3:
                test_data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32) * 0.1
            else:
                logger.error(f"{language} TensorFlow model has unexpected input shape: {input_shape}")
                return False

            try:
                pred_prob = model.predict(test_data, verbose=0)
                logger.info(f"{language} TensorFlow validation passed. Output shape: {pred_prob.shape}")
                return True
            except Exception as pred_error:
                logger.error(f"{language} TensorFlow prediction test failed: {pred_error}")
                return False
                
        except Exception as e:
            logger.error(f"{language} TensorFlow validation failed: {e}")
            return False
    
    def extract_landmarks_from_frame(self, image_bgr, mirror_mode=None):
        """Extract landmarks - IDENTICAL to camera_test.py"""
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                logger.debug("No hands detected")
                return np.zeros(126, dtype=np.float32)
            
            landmarks_flat = []
            hand_data = []
            
            # Process detected hands
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                wrist_x = hand_landmarks.landmark[0].x
                hand_data.append({
                    'landmarks': hand_landmarks,
                    'wrist_x': wrist_x,
                    'index': i
                })
            
            num_hands = len(hand_data)
            logger.debug(f"Detected {num_hands} hands")
            
            # Sort hands by position (left to right)
            if num_hands >= 2:
                hand_data.sort(key=lambda x: x['wrist_x'])
            
            # Extract landmarks for up to 2 hands
            for hand_idx in range(2):
                if hand_idx < len(hand_data):
                    hand_landmarks = hand_data[hand_idx]['landmarks']
                    for landmark in hand_landmarks.landmark:
                        landmarks_flat.extend([
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z)
                        ])
                else:
                    landmarks_flat.extend([0.0] * 63)
            
            # Ensure exactly 126 values
            if len(landmarks_flat) != 126:
                landmarks_flat = landmarks_flat[:126]
                while len(landmarks_flat) < 126:
                    landmarks_flat.append(0.0)
            
            return np.array(landmarks_flat, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Landmark extraction error: {e}")
            return None
    
    def extract_features_from_landmarks(self, landmarks_data):
        """Extract features - IDENTICAL to camera_test.py"""
        try:
            if extract_features is not None:
                # Use same pipeline as training
                landmark_cols = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']]
                df_landmarks = pd.DataFrame([landmarks_data], columns=landmark_cols)
                
                # Extract features without selection (use all features)
                features_df = extract_features(df_landmarks, perform_selection=False)
                
                if not features_df.empty:
                    # Remove label columns
                    features_for_prediction = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
                    return features_for_prediction
            
            # Fallback: create basic features
            return self.create_basic_features(landmarks_data)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return self.create_basic_features(landmarks_data)
    
    def create_basic_features(self, landmarks_data):
        """Create basic features - IDENTICAL to camera_test.py"""
        try:
            features = {}
            
            # Split into 2 hands
            hand1_data = landmarks_data[:63]
            hand2_data = landmarks_data[63:]
            
            for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
                if len(hand_data) >= 63:
                    x_coords = hand_data[::3]
                    y_coords = hand_data[1::3]
                    z_coords = hand_data[2::3]
                    
                    # Basic statistics
                    features[f'h{hand_idx}_x_mean'] = np.mean(x_coords)
                    features[f'h{hand_idx}_y_mean'] = np.mean(y_coords)
                    features[f'h{hand_idx}_z_mean'] = np.mean(z_coords)
                    features[f'h{hand_idx}_x_std'] = np.std(x_coords)
                    features[f'h{hand_idx}_y_std'] = np.std(y_coords)
                    features[f'h{hand_idx}_z_std'] = np.std(z_coords)
                    
                    # Distances
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    for tip_idx in [4, 8, 12, 16, 20]:  # fingertips
                        if tip_idx < len(x_coords):
                            dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                         (y_coords[tip_idx] - wrist_y)**2 + 
                                         (z_coords[tip_idx] - wrist_z)**2)
                            features[f'h{hand_idx}_tip_{tip_idx}_dist'] = dist
                else:
                    # Fill with zeros for missing hand
                    for stat in ['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std']:
                        features[f'h{hand_idx}_{stat}'] = 0.0
                    for tip_idx in [4, 8, 12, 16, 20]:
                        features[f'h{hand_idx}_tip_{tip_idx}_dist'] = 0.0
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Basic feature creation error: {e}")
            # Ultimate fallback
            basic_features = {f'feature_{i}': 0.0 for i in range(50)}
            return pd.DataFrame([basic_features])
    
    def validate_sklearn_model(self, model_data, language):
        """Validate sklearn model"""
        try:
            model = model_data.get('model')
            if model is None:
                return False
            
            # Get expected features
            n_features = getattr(model, 'n_features_in_', None)
            if n_features is None:
                if 'feature_names' in model_data:
                    n_features = len(model_data['feature_names'])
                else:
                    n_features = 126
            
            # Test prediction
            test_data = np.random.rand(1, n_features).astype(np.float32) * 0.1
            pred = model.predict(test_data)[0]
            logger.info(f"{language} sklearn validation passed: {pred}")
            return True
        except Exception as e:
            logger.error(f"{language} sklearn validation failed: {e}")
            return False
    
    def align_features_to_model(self, features_df, expected_feature_names, model_name):
        """Align input features to match model's expected feature order"""
        try:
            if not expected_feature_names:
                logger.warning(f"{model_name}: No expected feature names - using features as-is")
                return features_df
            
            current_cols = set(features_df.columns)
            expected_cols = set(expected_feature_names)
            
            # Add missing features with zeros
            missing = expected_cols - current_cols
            if missing:
                logger.info(f"{model_name}: Adding {len(missing)} missing features")
                for col in missing:
                    features_df[col] = 0.0
            
            # Remove extra features
            extra = current_cols - expected_cols
            if extra:
                logger.info(f"{model_name}: Removing {len(extra)} extra features")
                features_df = features_df.drop(columns=list(extra))
            
            # CRITICAL: Reorder columns to match training order
            features_df = features_df[expected_feature_names]
            
            logger.info(f"{model_name}: Features aligned - {len(expected_feature_names)} features in correct order")
            
            return features_df
            
        except Exception as e:
            logger.error(f"{model_name}: Feature alignment failed - {e}")
            return features_df
    
    def predict_tensorflow(self, features_df, model, meta):
        """Predict using TensorFlow model with proper feature alignment"""
        try:
            scaler = meta.get('scaler')
            label_encoder = meta.get('label_encoder')
            feature_names = meta.get('feature_names')
            
            if model is None:
                raise ValueError("TensorFlow model not found")

            # CRITICAL: Align features to model's expected order
            if feature_names:
                features_df = self.align_features_to_model(features_df, feature_names, "TensorFlow")
            else:
                logger.warning("TensorFlow: No feature_names found - predictions may be inaccurate")
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
                logger.debug("TensorFlow: Features scaled")
            else:
                features_scaled = features_df.values
                logger.warning("TensorFlow: No scaler found")
            
            # Final input shape validation
            expected_features = model.input_shape[1]
            if features_scaled.shape[1] != expected_features:
                logger.error(f"TensorFlow: Shape mismatch! Expected {expected_features}, got {features_scaled.shape[1]}")
                if features_scaled.shape[1] < expected_features:
                    padding = np.zeros((features_scaled.shape[0], expected_features - features_scaled.shape[1]))
                    features_scaled = np.hstack((features_scaled, padding))
                    logger.warning(f"TensorFlow: Padded to {expected_features} features")
                elif features_scaled.shape[1] > expected_features:
                    features_scaled = features_scaled[:, :expected_features]
                    logger.warning(f"TensorFlow: Truncated to {expected_features} features")
            
            # Make prediction
            predictions_prob = model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(predictions_prob, axis=1)[0]
            confidence = np.max(predictions_prob)
            
            # Convert to label
            if label_encoder:
                prediction = label_encoder.inverse_transform([predicted_class_idx])[0]
                logger.debug(f"TensorFlow: Class {predicted_class_idx} -> {prediction}")
            else:
                prediction = f"Class_{predicted_class_idx}"
                logger.warning("TensorFlow: No label_encoder found")
            
            logger.info(f"TensorFlow: Prediction={prediction}, Confidence={confidence:.3f}")
            return str(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"TensorFlow prediction error: {e}")
            return None, 0.0
    
    def predict_sklearn(self, features_df, model_data):
        """Predict using sklearn model"""
        try:
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            feature_names = model_data.get('feature_names')
            
            if model is None:
                raise ValueError("Model not found")

            # Handle feature alignment
            if feature_names:
                current_cols = set(features_df.columns)
                expected_cols = set(feature_names)

                # Add missing columns
                missing = expected_cols - current_cols
                if missing:
                    for col in missing:
                        features_df[col] = 0.0
                
                # Remove extra columns
                extra = current_cols - expected_cols
                if extra:
                    features_df = features_df.drop(columns=list(extra))

                # Reorder columns
                features_df = features_df[feature_names]
                logger.debug(f"Features aligned to {len(feature_names)} expected features")
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Get confidence
            confidence = 0.7  # Default
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = np.max(probabilities)
                elif hasattr(model, 'decision_function'):
                    decision_values = model.decision_function(features_scaled)[0]
                    if isinstance(decision_values, np.ndarray):
                        confidence = np.max(np.abs(decision_values)) / 10.0
                    else:
                        confidence = abs(decision_values) / 10.0
                    confidence = min(1.0, max(0.1, confidence))
            except Exception:
                confidence = 0.7
            
            return str(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Sklearn prediction error: {e}")
            return None, 0.0
    
    def predict_with_model(self, features_df, language):
        """Make prediction using best available model"""
        try:
            model_info = self.models.get(language.upper())
            if not model_info:
                logger.warning(f"No model info for {language.upper()}")
                return None, 0.0
            
            # Try TensorFlow first if available (usually more accurate)
            if 'tensorflow' in model_info['available_models']:
                logger.debug(f"Using TensorFlow model for {language.upper()}")
                return self.predict_tensorflow(features_df, model_info['tensorflow_model'], model_info['tensorflow_meta'])
            elif 'sklearn' in model_info['available_models']:
                logger.debug(f"Using sklearn model for {language.upper()}")
                return self.predict_sklearn(features_df, model_info['sklearn_model'])
            else:
                logger.warning(f"No usable models for {language.upper()}")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Model prediction dispatch failed for {language}: {e}")
            return None, 0.0
    
    def predict_sign(self, image_bgr, language_type='bisindo', mirror_mode=None):
        """Main prediction function - IDENTICAL preprocessing as camera_test.py"""
        try:
            language_type_upper = language_type.upper()
            logger.info(f"Predicting for {language_type_upper}")
            
            if language_type_upper not in self.models:
                available = list(self.models.keys())
                error_msg = f"Model not available for {language_type_upper}. Available: {available}"
                logger.error(error_msg)
                return "Demo_A", 0.7, error_msg
            
            # Extract landmarks - IDENTICAL to camera_test.py
            landmarks_np = self.extract_landmarks_from_frame(image_bgr, mirror_mode=mirror_mode)
            if landmarks_np is None:
                logger.warning("Landmark extraction failed")
                return "No hand detected", 0.0, "Landmark extraction failed"
            
            if np.all(landmarks_np == 0.0):
                logger.warning("No hands detected")
                return "No hand detected", 0.0, "No hands detected"
            
            # Extract features - IDENTICAL to camera_test.py
            features_df = self.extract_features_from_landmarks(landmarks_np)
            
            if features_df.empty:
                logger.error("Feature creation failed")
                return "Feature error", 0.0, "Feature creation failed"
            
            # Make prediction
            model_info = self.models[language_type_upper]
            prediction, confidence = self.predict_with_model(features_df, language_type_upper)
                
            if prediction is None:
                return "Prediction failed", 0.0, "Model prediction failed"
            
            # Check for valid hand detection
            hand1_exists = not np.all(landmarks_np[0:63] == 0.0)
            hand2_exists = not np.all(landmarks_np[63:126] == 0.0)
            hands_detected = sum([hand1_exists, hand2_exists])
            
            logger.info(f"SUCCESS: {prediction} (confidence: {confidence:.3f}, hands: {hands_detected})")
            
            return prediction, confidence, f"Success - {hands_detected} hand(s) detected"
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0, f"Prediction error: {str(e)}"


# Initialize API
api = FixedSignLanguageAPI()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with comprehensive status"""
    try:
        models_status = {}
        for lang, info in api.models.items():
            models_status[lang] = {
                'sklearn': 'loaded' if 'sklearn_model' in info else 'not_loaded',
                'tensorflow': 'loaded' if 'tensorflow_model' in info else 'not_loaded',
                'available_types': info['available_models']
            }

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_summary': models_status,
            'total_languages_loaded': len(api.models),
            'mediapipe_ready': api.hands is not None,
            'feature_extractor_available': extract_features_available,
            'backend_version': '2.1.0-fixed-preprocessing',
            'endpoints': {
                'health': '/api/health',
                'models': '/api/models', 
                'translate': '/api/translate'
            }
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'backend_alive': True
        }), 500

@app.route('/api/translate', methods=['POST'])
def translate_sign_endpoint():
    """Fixed translate endpoint with identical preprocessing"""
    try:
        logger.info("Translate endpoint called")
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        if not api.models:
            error_msg = 'No models loaded'
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg, 'prediction': 'Demo'}), 500
            
        image_data_b64 = data['image']
        language_type = data.get('language_type', 'bisindo').lower()
        mirror_mode = data.get('mirror_mode', None)

        # Decode image
        try:
            if ',' in image_data_b64:
                image_data_b64 = image_data_b64.split(',')[1]
            
            image_bytes = base64.b64decode(image_data_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                raise ValueError("Could not decode image")
                
            if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
                raise ValueError("Invalid image dimensions")
            
            logger.info(f"Decoded image shape: {image_bgr.shape}")
                
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return jsonify({'success': False, 'error': f'Image decoding failed: {e}'}), 400
            
        # Make prediction
        prediction, confidence, message = api.predict_sign(image_bgr, language_type, mirror_mode=mirror_mode)
        
        response = {
            'success': prediction is not None and prediction not in ["No hand detected", "Feature error", "Error"],
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
        return jsonify({'success': False, 'error': str(e), 'prediction': 'Error'}), 500

@app.route('/api/models', methods=['GET'])
def get_models_endpoint():
    """Models info endpoint"""
    return jsonify({
        'available_models': list(api.models.keys()),
        'total_models': len(api.models),
        'models_detail': {
            lang: {
                'available_types': info['available_models'],
                'sklearn_feature_count': len(info.get('sklearn_model', {}).get('feature_names', [])) if 'sklearn_model' in info else 0,
                'tensorflow_feature_count': len(info.get('tensorflow_meta', {}).get('feature_names', [])) if 'tensorflow_meta' in info else 0
            } 
            for lang, info in api.models.items()
        }
    })

@app.before_request
def handle_preflight():
    """Handle CORS preflight"""
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.after_request
def after_request(response):
    """Add CORS headers"""
    origin = request.headers.get('Origin')
    
    if origin:
        for allowed_origin in CORS_ORIGINS:
            if (
                allowed_origin == '*' or
                origin == allowed_origin or
                (allowed_origin.startswith('https://*.') and origin.endswith(allowed_origin[8:]))
            ):
                response.headers['Access-Control-Allow-Origin'] = origin
                break
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,User-Agent,Accept,Origin,X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'false'
    
    return response

@app.route("/")
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Fixed Sign Language API is running",
        "available_endpoints": [
            "/api/translate",
            "/api/models", 
            "/api/health"
        ]
    })

if __name__ == '__main__':
    print("\nFIXED SIGN LANGUAGE API")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Models loaded: {list(api.models.keys())}")
    print(f"Total models: {len(api.models)}")
    print(f"MediaPipe ready: {api.hands is not None}")
    print(f"Feature extractor available: {extract_features_available}")
    
    if not api.models:
        print("\nNO MODELS LOADED - API WILL USE DEMO PREDICTIONS")
    else:
        print("\nAPI READY FOR PREDICTIONS")
        for lang, info in api.models.items():
            available_types = ', '.join(info['available_models'])
            print(f"    {lang}: {available_types}")
    
    print("=" * 50)
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Server start failed: {e}")
        sys.exit(1)
