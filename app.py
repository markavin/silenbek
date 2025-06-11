#!/usr/bin/env python3
# app.py - FIXED VERSION - Clean Model Loading & Prediction

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
        "data/models/sign_language_model_bisindo_feature_names.pkl": "https://drive.google.com/uc?export=download&id=1mZuNshrTfJdeBcPZdDLFxsIiaGvm5zOV",
        "data/models/sign_language_model_bisindo_sklearn.pkl": "https://drive.google.com/uc?export=download&id=1KiVbDoeEhi8e_ALg3MTjnSYcVJCcsnab",
        "data/models/sign_language_model_bisindo_tensorflow_meta.pkl": "https://drive.google.com/uc?export=download&id=1UtvlHe0tMlQl0JxdrRtXAiHD9dfJ_r4u",
        "data/models/sign_language_model_bisindo_tensorflow.h5": "https://drive.google.com/uc?export=download&id=16bs2DwGp5nOvlvutny7f-oBWLzi00Srd",
        "data/models/sign_language_model_bisindo.pkl": "https://drive.google.com/uc?export=download&id=1GNqTOAWwoGRyd4n00zv17d4YfkByS0tA",
        "data/models/sign_language_model_sibi_feature_names.pkl": "https://drive.google.com/uc?export=download&id=16XCooZ4DWpKGTy3yEVrGRkEamqpreIsa",
        "data/models/sign_language_model_sibi_sklearn.pkl": "https://drive.google.com/uc?export=download&id=1r5fDK7blBHM-CXHWs3NjKy14RZ4B_DpG",
        "data/models/sign_language_model_sibi_tensorflow_meta.pkl": "https://drive.google.com/uc?export=download&id=1ujlwbf5KZaeV3hdo8-u7jhRIRtX4MbtK",
        "data/models/sign_language_model_sibi_tensorflow.h5": "https://drive.google.com/uc?export=download&id=1Zz55YonwMcWsR76CW1W5zBW6uzzuaZDh",
        "data/models/sign_language_model_sibi.pkl": "https://drive.google.com/uc?export=download&id=1UGGPtCgiQzfdV4CdN3Cg2o3V0-BUlpZj"
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
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4
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
                        compile=False  # Skip compilation to avoid compatibility issues
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
                    # Try loading with different compatibility options
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

            # Test prediction with error handling
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
        """Extract landmarks with proper error handling"""
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
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                wrist_x = hand_landmarks.landmark[0].x
                
                # Adjust handedness based on mirror mode
                if mirror_mode is True:
                    adjusted_label = "Left" if hand_label == "Right" else "Right"
                elif mirror_mode is False:
                    adjusted_label = hand_label
                else:
                    adjusted_label = hand_label
                
                hand_data.append({
                    'landmarks': hand_landmarks,
                    'adjusted_label': adjusted_label,
                    'score': hand_score,
                    'wrist_x': wrist_x,
                    'index': i
                })
            
            num_hands = len(hand_data)
            logger.debug(f"Detected {num_hands} hands")
            
            # Sort hands
            if num_hands >= 2:
                hand_data.sort(key=lambda x: x['wrist_x'])
            else:
                hand_data.sort(key=lambda x: -x['score'])
            
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
            
            # Debug: Log first few feature values
            logger.debug(f"{model_name}: First 5 features: {dict(list(features_df.iloc[0].head().items()))}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"{model_name}: Feature alignment failed - {e}")
            return features_df
        """Create features from landmarks with proper structure"""
        try:
            features = {}
            
            # Split into two hands
            hand1_data = landmarks_data[:63]
            hand2_data = landmarks_data[63:]
            
            feature_idx = 0
            
            for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
                if len(hand_data) >= 63:
                    x_coords = hand_data[::3]
                    y_coords = hand_data[1::3]
                    z_coords = hand_data[2::3]
                    
                    # Check if hand exists
                    hand_exists = not all(x == 0 and y == 0 for x, y in zip(x_coords, y_coords))
                    
                    if hand_exists:
                        # Basic statistics
                        stats = [
                            np.mean(x_coords), np.std(x_coords), np.min(x_coords), np.max(x_coords),
                            np.mean(y_coords), np.std(y_coords), np.min(y_coords), np.max(y_coords),
                            np.mean(z_coords), np.std(z_coords), np.min(z_coords), np.max(z_coords),
                        ]
                        
                        for stat in stats:
                            features[f'feature_{feature_idx}'] = float(stat) if np.isfinite(stat) else 0.0
                            feature_idx += 1
                        
                        # Distances from wrist to fingertips
                        wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                        fingertip_indices = [4, 8, 12, 16, 20]
                        
                        for tip_idx in fingertip_indices:
                            if tip_idx < len(x_coords):
                                dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                             (y_coords[tip_idx] - wrist_y)**2 + 
                                             (z_coords[tip_idx] - wrist_z)**2)
                                features[f'feature_{feature_idx}'] = float(dist) if np.isfinite(dist) else 0.0
                                feature_idx += 1
                        
                        # Hand geometry
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        features[f'feature_{feature_idx}'] = float(width) if np.isfinite(width) else 0.0
                        feature_idx += 1
                        features[f'feature_{feature_idx}'] = float(height) if np.isfinite(height) else 0.0
                        feature_idx += 1
                    else:
                        # Fill zeros for missing hand
                        for _ in range(19):  # 12 stats + 5 distances + 2 geometry
                            features[f'feature_{feature_idx}'] = 0.0
                            feature_idx += 1
                else:
                    # Hand data incomplete
                    for _ in range(19):
                        features[f'feature_{feature_idx}'] = 0.0
                        feature_idx += 1
            
            # Two-hand interaction features
            if len(hand1_data) >= 63 and len(hand2_data) >= 63:
                hand1_exists = not all(x == 0 and y == 0 for x, y in zip(hand1_data[::3], hand1_data[1::3]))
                hand2_exists = not all(x == 0 and y == 0 for x, y in zip(hand2_data[::3], hand2_data[1::3]))
                
                if hand1_exists and hand2_exists:
                    wrist1_x, wrist1_y = hand1_data[0], hand1_data[1]
                    wrist2_x, wrist2_y = hand2_data[0], hand2_data[1]
                    
                    inter_wrist_dist = np.sqrt((wrist1_x - wrist2_x)**2 + (wrist1_y - wrist2_y)**2)
                    features[f'feature_{feature_idx}'] = float(inter_wrist_dist) if np.isfinite(inter_wrist_dist) else 0.0
                    feature_idx += 1
                    
                    features[f'feature_{feature_idx}'] = float(wrist1_x - wrist2_x) if np.isfinite(wrist1_x - wrist2_x) else 0.0
                    feature_idx += 1
                    features[f'feature_{feature_idx}'] = float(wrist1_y - wrist2_y) if np.isfinite(wrist1_y - wrist2_y) else 0.0
                    feature_idx += 1
                else:
                    for _ in range(3):
                        features[f'feature_{feature_idx}'] = 0.0
                        feature_idx += 1
            else:
                for _ in range(3):
                    features[f'feature_{feature_idx}'] = 0.0
                    feature_idx += 1
            
            # Pad to minimum expected features
            while feature_idx < 80:
                features[f'feature_{feature_idx}'] = 0.0
                feature_idx += 1
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            basic_features = {f'feature_{i}': 0.0 for i in range(80)}
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

            # Test prediction with error handling
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
        """Predict using sklearn model with proper feature handling"""
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
                        confidence = np.max(np.abs(decision_values)) / 10.0  # Normalize
                    else:
                        confidence = abs(decision_values) / 10.0
                    confidence = min(1.0, max(0.1, confidence))
            except Exception:
                confidence = 0.7
            
            return str(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Sklearn prediction error: {e}")
            return None, 0.0
    
    def predict_sign(self, image_bgr, language_type='bisindo', mirror_mode=None):
        """Main prediction with fixed feature processing"""
        try:
            language_type_upper = language_type.upper()
            logger.info(f"Predicting for {language_type_upper}")
            
            if language_type_upper not in self.models:
                available = list(self.models.keys())
                error_msg = f"Model not available for {language_type_upper}. Available: {available}"
                logger.error(error_msg)
                return "Demo_A", 0.7, error_msg
            
            # Extract landmarks
            landmarks_np = self.extract_landmarks_from_frame(image_bgr, mirror_mode=mirror_mode)
            if landmarks_np is None:
                logger.warning("Landmark extraction failed")
                return "No hand detected", 0.0, "Landmark extraction failed"
            
            if np.all(landmarks_np == 0.0):
                logger.warning("No hands detected")
                return "No hand detected", 0.0, "No hands detected"
            
            # Create features
            features_df = self.create_features_from_landmarks(landmarks_np)
            
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
    """Enhanced health check with feature names debugging"""
    try:
        models_status = {}
        for lang, info in api.models.items():
            models_status[lang] = {
                'sklearn': 'loaded' if 'sklearn_model' in info else 'not_loaded',
                'tensorflow': 'loaded' if 'tensorflow_model' in info else 'not_loaded',
                'available_types': info['available_models']
            }
            
            # Add feature names info for debugging
            if 'sklearn_model' in info:
                sklearn_feature_names = info['sklearn_model'].get('feature_names', [])
                models_status[lang]['sklearn_features'] = {
                    'count': len(sklearn_feature_names) if sklearn_feature_names else 0,
                    'first_5': sklearn_feature_names[:5] if sklearn_feature_names else [],
                    'has_feature_names': bool(sklearn_feature_names)
                }
            
            if 'tensorflow_meta' in info:
                tf_feature_names = info['tensorflow_meta'].get('feature_names', [])
                models_status[lang]['tensorflow_features'] = {
                    'count': len(tf_feature_names) if tf_feature_names else 0,
                    'first_5': tf_feature_names[:5] if tf_feature_names else [],
                    'has_feature_names': bool(tf_feature_names)
                }

        # Test feature extraction
        try:
            from src.data_preprocessing.feature_extractor import get_expected_feature_names
            expected_features = get_expected_feature_names()
            feature_test_result = {
                'feature_extractor_working': True,
                'expected_feature_count': len(expected_features),
                'first_5_expected': expected_features[:5],
                'last_5_expected': expected_features[-5:]
            }
        except Exception as feat_e:
            feature_test_result = {
                'feature_extractor_working': False,
                'error': str(feat_e)
            }

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_summary': models_status,
            'total_languages_loaded': len(api.models),
            'mediapipe_ready': api.hands is not None,
            'feature_extractor_available': extract_features_available,
            'feature_test': feature_test_result,
            'backend_version': '2.0.3-feature-fixed',
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
    """Fixed translate endpoint"""
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

@app.route('/api/debug/features', methods=['GET'])
def debug_features():
    """Debug endpoint to test feature extraction consistency"""
    try:
        # Create dummy landmarks for testing
        dummy_landmarks = np.random.rand(126).astype(np.float32) * 0.5
        
        # Test feature creation
        features_df = api.create_features_from_landmarks(dummy_landmarks)
        
        if features_df.empty:
            return jsonify({
                'success': False,
                'error': 'Feature creation failed'
            })
        
        feature_columns = [col for col in features_df.columns 
                          if col not in ['label', 'sign_language_type', 'is_mirrored', 'image_name']]
        
        # Get expected features
        try:
            from src.data_preprocessing.feature_extractor import get_expected_feature_names, validate_feature_names_consistency
            expected_features = get_expected_feature_names()
            validation_result = validate_feature_names_consistency(features_df)
        except ImportError:
            expected_features = []
            validation_result = {'error': 'Cannot import feature validation'}
        
        # Compare with model expectations
        model_comparisons = {}
        for lang, model_info in api.models.items():
            if 'sklearn_model' in model_info:
                sklearn_features = model_info['sklearn_model'].get('feature_names', [])
                model_comparisons[f'{lang}_sklearn'] = {
                    'model_feature_count': len(sklearn_features),
                    'extracted_feature_count': len(feature_columns),
                    'feature_match': set(sklearn_features) == set(feature_columns) if sklearn_features else False,
                    'missing_in_extracted': list(set(sklearn_features) - set(feature_columns)) if sklearn_features else [],
                    'extra_in_extracted': list(set(feature_columns) - set(sklearn_features)) if sklearn_features else []
                }
            
            if 'tensorflow_meta' in model_info:
                tf_features = model_info['tensorflow_meta'].get('feature_names', [])
                model_comparisons[f'{lang}_tensorflow'] = {
                    'model_feature_count': len(tf_features),
                    'extracted_feature_count': len(feature_columns),
                    'feature_match': set(tf_features) == set(feature_columns) if tf_features else False,
                    'missing_in_extracted': list(set(tf_features) - set(feature_columns)) if tf_features else [],
                    'extra_in_extracted': list(set(feature_columns) - set(tf_features)) if tf_features else []
                }
        
        return jsonify({
            'success': True,
            'extracted_features': {
                'count': len(feature_columns),
                'first_10': feature_columns[:10],
                'last_10': feature_columns[-10:]
            },
            'expected_features': {
                'count': len(expected_features),
                'first_10': expected_features[:10] if expected_features else [],
                'last_10': expected_features[-10:] if expected_features else []
            },
            'validation_result': validation_result,
            'model_comparisons': model_comparisons,
            'recommendations': [
                "If feature_match is False, models will predict incorrectly",
                "Missing features should be added with default values",
                "Extra features should be removed or model retrained",
                "Feature order must match training exactly"
            ]
        })
        
    except Exception as e:
        logger.error(f"Feature debug failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/models', methods=['GET'])
def get_models_endpoint():
    """Enhanced models info with feature debugging"""
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
        },
        'debug_endpoint': '/api/debug/features'
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
