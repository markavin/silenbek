#!/usr/bin/env python3
# app.py - Enhanced for Two-Hand Detection - FIXED VERSION

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

# Fix for Windows console output (if running locally on Windows)
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

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
    logger.info("Feature extractor imported successfully from src.data_preprocessing")
except ImportError as e:
    extract_features_available = False
    logger.error(f"❌ Feature extractor not available: {e}.")

app = Flask(__name__)

HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))

CORS_ORIGINS = os.environ.get('FRONTEND_URL', '*').split(',')
CORS_RESOURCES = {r"/*": {"origins": CORS_ORIGINS}}
CORS(app, resources=CORS_RESOURCES, methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type', 'Authorization', 'User-Agent'])
logger.info(f"🔓 CORS configured for origins: {CORS_ORIGINS}")

# NEW: Download model files from Google Drive with validation

def download_model_files():
    model_urls = {
        "data/models/sign_language_model_bisindo_sklearn.pkl": "https://drive.google.com/uc?export=download&id=1GbkAytkTb4gEbjL7NnTKTDCWwT4z10Vr",
        "data/models/sign_language_model_bisindo_tensorflow_meta.pkl": "https://drive.google.com/uc?export=download&id=1dckHPCKeXXGwPhZxxtqllb_VOpgF6Dnl",
        "data/models/sign_language_model_bisindo_tensorflow.h5": "https://drive.google.com/uc?export=download&id=1hS_27-0oprjLFIxOFFLKt2CVdYDfdJDV",
        "data/models/sign_language_model_bisindo.pkl": "https://drive.google.com/uc?export=download&id=1BZwJ92gT-3EkGT0ZEDEXvXIn0FDcg_Z0",

        "data/models/sign_language_model_sibi_sklearn.pkl": "https://drive.google.com/uc?export=download&id=12XNi2oH59KwMs915pBJHEMrpHIr3ykHw",
        "data/models/sign_language_model_sibi_tensorflow_meta.pkl": "https://drive.google.com/uc?export=download&id=1G0vEMqU9lTL0Ks9FC0x2edDvqN3MQREs",
        "data/models/sign_language_model_sibi_tensorflow.h5": "https://drive.google.com/uc?export=download&id=1J8ZR19ejQOXgPab-6zw4tIHL0GHlz-nJ",
        "data/models/sign_language_model_sibi.pkl": "https://drive.google.com/uc?export=download&id=1pJ9-2ucfd7hUw6PSJV9xWwZ1zLwEstK1"
    }

    for local_path, url in model_urls.items():
        if not os.path.exists(local_path):
            logger.info(f"📥 Downloading model file to {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                response = requests.get(url)
                content_type = response.headers.get('Content-Type', '')
                if response.status_code == 200 and 'text/html' not in content_type.lower() and b"<html" not in response.content[:100].lower():
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ File saved: {local_path}")
                else:
                    logger.error(f"❌ Invalid file content for {url}. Content-Type: {content_type}. File not saved.")
            except Exception as e:
                logger.error(f"❌ Failed to download {url}: {e}")

# Run download before initializing the API
download_model_files()

class EnhancedSignLanguageAPI:
    def __init__(self): 
        self.models = {}
        self.project_root = project_root
        
        # Enhanced MediaPipe setup for better two-hand detection
        self.mp_hands = mp.solutions.hands
        # Menggunakan konteks manager 'with' untuk inisialisasi Hands
        # Ini akan memastikan sumber daya dilepaskan dengan benar.
        # Namun, untuk kelas, Anda harus menginisialisasinya di _init_
        # dan menutupnya secara manual jika diperlukan, atau mengandalkan garbage collection.
        # Untuk kasus Flask API yang request-per-request, bisa juga inisialisasi di dalam fungsi
        # predict_sign jika overhead tidak masalah. Untuk saat ini, biarkan di _init_.
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,  # Ensure 2 hands
            min_detection_confidence=0.5, # Lower threshold for better detection
            min_tracking_confidence=0.4 # Lower threshold for tracking (used in video, but good for static too)
        )
        
        self.load_models()
        
    def load_models(self):
        """Load available models"""
        logger.info("Loading models...")
        
        # Define base model path relative to project_root
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
            
            # Load sklearn model (prefer this for stability)
            if config['sklearn_path'].exists():
                try:
                    sklearn_data = joblib.load(config['sklearn_path'])
                    # Validate that it's a dict with 'model' key
                    if isinstance(sklearn_data, dict) and 'model' in sklearn_data:
                        if self.validate_sklearn_model(sklearn_data, config['name']):
                            model_info['sklearn_model'] = sklearn_data
                            model_info['available_models'].append('sklearn')
                            logger.info(f"  {config['name']}: Scikit-learn model loaded from {config['sklearn_path']}")
                        else:
                            logger.warning(f"  {config['name']}: Scikit-learn model validation failed.")
                    else:
                        logger.warning(f"  {config['name']}: Scikit-learn model data is not in expected dictionary format.")
                except Exception as e:
                    logger.warning(f"  {config['name']}: Scikit-learn load failed from {config['sklearn_path']} - {e}")
            else:
                logger.info(f"  {config['name']}: Scikit-learn model not found at {config['sklearn_path']}")

            # Load TensorFlow model
            if config['tensorflow_path'].exists() and config['tensorflow_meta_path'].exists():
                try:
                    import tensorflow as tf # Import TensorFlow inside this block
                    tf_model = tf.keras.models.load_model(str(config['tensorflow_path'])) # Convert Path to str
                    tf_meta = joblib.load(config['tensorflow_meta_path'])
                    
                    if self.validate_tensorflow_model(tf_model, tf_meta, config['name']):
                        model_info['tensorflow_model'] = tf_model
                        model_info['tensorflow_meta'] = tf_meta
                        model_info['available_models'].append('tensorflow')
                        logger.info(f"  {config['name']}: TensorFlow model loaded from {config['tensorflow_path']}")
                    else:
                        logger.warning(f"  {config['name']}: TensorFlow model validation failed.")
                except ImportError:
                    logger.warning(f"  {config['name']}: TensorFlow library not available. Cannot load TensorFlow model.")
                except Exception as e:
                    logger.warning(f"  {config['name']}: TensorFlow load failed from {config['tensorflow_path']} - {e}")
            else:
                logger.info(f"  {config['name']}: TensorFlow model or meta not found at {config['tensorflow_path']}")
            
            if model_info['available_models']:
                self.models[config['name']] = model_info
        
        logger.info(f"Loaded {len(self.models)} language models: {list(self.models.keys())}")
    
    def validate_sklearn_model(self, model_data, language):
        """Validate sklearn model by trying a dummy prediction"""
        try:
            model = model_data.get('model')
            if model is None:
                logger.error(f"{language} sklearn model data is missing 'model' key.")
                return False
            
            # Use a more robust check for n_features_in_
            n_features = getattr(model, 'n_features_in_', None)
            if n_features is None:
                # Fallback for older sklearn versions or custom models
                # Try to get number of features from a saved feature_names list in meta if available
                if 'feature_names' in model_data and model_data['feature_names']:
                    n_features = len(model_data['feature_names'])
                else:
                    n_features = 126 # Default to landmarks flat size if no specific features expected
            
            if n_features > 0:
                test_data = np.random.rand(1, n_features).astype(np.float32) * 0.1
                pred = model.predict(test_data)[0]
                logger.info(f"{language} sklearn dummy prediction successful: {pred}")
                return True
            else:
                logger.error(f"{language} sklearn model has invalid n_features_in_ or features not defined.")
                return False
        except Exception as e:
            logger.error(f"{language} sklearn validation failed: {e}")
            return False
    
    def validate_tensorflow_model(self, model, meta, language):
        """Validate TensorFlow model by trying a dummy prediction"""
        try:
            input_shape = model.input_shape[1:] # Get shape excluding batch dimension
            
            # Create dummy data matching expected input shape
            if len(input_shape) == 1: # Flattened input (e.g., (None, 126))
                test_data = np.random.rand(1, input_shape[0]).astype(np.float32) * 0.1
            elif len(input_shape) == 3: # Image input (e.g., (None, 64, 64, 3))
                test_data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32) * 0.1
            else:
                logger.error(f"{language} TensorFlow model has unexpected input shape: {input_shape}")
                return False

            pred_prob = model.predict(test_data, verbose=0)
            logger.info(f"{language} TensorFlow dummy prediction successful. Output shape: {pred_prob.shape}")
            return True
        except Exception as e:
            logger.error(f"{language} TensorFlow validation failed: {e}")
            return False
    
    def preprocess_image_for_detection(self, image_bgr):
        """Enhanced preprocessing for better hand detection (MediaPipe)"""
        try:
            height, width = image_bgr.shape[:2]
            
            # Resize to optimal size for MediaPipe (e.g., 640px on longest side)
            target_size = 640
            if width > height:
                new_width = target_size
                new_height = int(height * target_size / width)
            else:
                new_height = target_size
                new_width = int(width * target_size / height)
            
            # Use cv2.INTER_AREA for shrinking, cv2.INTER_LINEAR/CUBIC for enlarging
            # resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # Use INTER_LINEAR for general good quality resizing, safer for mixed use
            resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # --- Enhanced preprocessing for better hand detection (adjust as needed) ---
            # 1. Improve contrast (CLAHE) - often helps in varying lighting
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Increased clipLimit from 2.0
            l_channel = clahe.apply(l_channel)
            enhanced = cv2.merge((l_channel, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Noise reduction while preserving hand edges (Bilateral Filter)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75) # Increased filter size and sigma values
            
            logger.debug(f"Image preprocessed for detection: {denoised.shape}")
            return denoised
            
        except Exception as e:
            logger.warning(f"Preprocessing for detection failed: {e}. Returning original image.")
            return image_bgr # Return original if preprocessing fails
    
    def extract_landmarks_from_frame(self, image_bgr, mirror_mode=None):
        """Enhanced landmark extraction with better two-hand support using MediaPipe Hands"""
        try:
            # Preprocess the image specifically for MediaPipe detection
            processed_image = self.preprocess_image_for_detection(image_bgr)
            
            # MediaPipe expects RGB
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                logger.debug("No hands detected by MediaPipe.")
                # Return empty list of landmarks for consistent feature extraction
                return np.zeros(126, dtype=np.float32) # Return a zero-filled array of expected size (2 hands * 21 landmarks * 3 coords)
            
            landmarks_flat = []
            hand_data = []
            
            # Process all detected hands
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label # "Left" or "Right"
                hand_score = handedness.classification[0].score
                wrist_x = hand_landmarks.landmark[0].x
                
                # Adjust handedness based on mirror mode (if camera is mirrored)
                # If mirror_mode is True, camera input is mirrored (e.g., selfie cam), so "Right" hand in image is actually user's Left.
                # If mirror_mode is False, camera input is not mirrored (e.g., external cam), so "Right" hand in image is user's Right.
                if mirror_mode is True: # User's perspective is mirrored
                    adjusted_label = "Left" if hand_label == "Right" else "Right" # Flip label
                elif mirror_mode is False: # User's perspective is not mirrored
                    adjusted_label = hand_label # Keep label as is
                else: # Default or unknown, keep original label
                    adjusted_label = hand_label
                
                hand_data.append({
                    'landmarks': hand_landmarks,
                    'original_label': hand_label,
                    'adjusted_label': adjusted_label, # The label we will use for sorting/assignment
                    'score': hand_score,
                    'wrist_x': wrist_x, # For spatial sorting
                    'index': i # Original index
                })
            
            num_hands = len(hand_data)
            hand_info = [(h['adjusted_label'], f"{h['score']:.2f}") for h in hand_data]
            logger.info(f"Detected {num_hands} hands by MediaPipe: {hand_info}")
            
            # --- Enhanced hand sorting for consistency ---
            # Sort detected hands:
            # For two-hand signs (BISINDO), sort by wrist X-position (left-to-right in image)
            # For one-hand signs (SIBI), sort by confidence (highest confidence first)
            
            # We assume BISINDO might use 2 hands, SIBI uses 1.
            # If language_type is 'bisindo', prioritize spatial sorting for 2 hands.
            # Otherwise, sort by confidence.
            # NOTE: Current logic sorts by confidence if less than 2 hands, else by X.
            # This is okay, but for true BISINDO, if 2 hands are detected, sort by X.
            # If only 1 hand for BISINDO, it's ambiguous.
            
            if num_hands >= 2:
                # Sort by X position (left hand (lower X) first for consistent order)
                # MediaPipe coordinates are 0-1, so smaller X means left side of image
                hand_data.sort(key=lambda x: x['wrist_x'])
                logger.debug("Sorted hands by X-position (left to right).")
            else: # Single hand or no hands, sort by confidence
                hand_data.sort(key=lambda x: -x['score']) # Descending score
                logger.debug("Sorted hands by confidence (highest first).")
            
            # Extract landmarks for up to 2 hands (always 21 landmarks per hand * 3 coords = 63)
            # Pad with zeros if less than 2 hands are detected, to maintain consistent input size (126 features).
            for hand_idx in range(2): # Always process for two hands
                if hand_idx < len(hand_data):
                    hand_landmarks = hand_data[hand_idx]['landmarks']
                    # hand_label = hand_data[hand_idx]['adjusted_label'] # Not used here, but good for logging
                    # hand_confidence = hand_data[hand_idx]['score'] # Not used here
                    
                    for landmark in hand_landmarks.landmark:
                        landmarks_flat.extend([
                            float(landmark.x),
                            float(landmark.y),
                            float(landmark.z)
                        ])
                else:
                    # Pad with zeros for missing hand (21 landmarks * 3 coords = 63 zeros)
                    logger.debug(f"Hand {hand_idx + 1} not detected, padding with zeros (63 values).")
                    landmarks_flat.extend([0.0] * 63)
            
            # Ensure exactly 126 values (2 hands * 21 landmarks * 3 coords)
            # This might happen if there's an unexpected issue with MediaPipe output
            if len(landmarks_flat) != 126:
                logger.warning(f"Extracted landmark count is {len(landmarks_flat)}, not 126. Truncating/Padding.")
                landmarks_flat = landmarks_flat[:126] # Truncate if too long
                while len(landmarks_flat) < 126:
                    landmarks_flat.append(0.0) # Pad if too short
            
            logger.info(f"Extracted {len(landmarks_flat)} landmark values for {num_hands} detected hands.")
            return np.array(landmarks_flat, dtype=np.float32) # Convert to numpy array
            
        except Exception as e:
            logger.error(f"Landmark extraction error: {e}")
            return None # Return None if extraction totally fails
    
    def extract_features_using_pipeline(self, landmarks_data_np, language):
        """Extract features using the same pipeline as training,
           falling back to basic features if pipeline not available."""
        try:
            if extract_features_available and landmarks_data_np is not None:
                # Convert numpy array to DataFrame for the feature extractor
                # Assuming 21 landmarks per hand, 3 coords (x,y,z)
                # Total 42 landmarks (if two hands), 126 coords.
                # If feature_extractor expects 'landmark_0_x' ... 'landmark_41_z'
                landmark_cols = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']]
                
                # Reshape if necessary (should be (1, 126) from extract_landmarks_from_frame)
                if landmarks_data_np.ndim == 1:
                    df_landmarks = pd.DataFrame([landmarks_data_np], columns=landmark_cols)
                elif landmarks_data_np.ndim == 2 and landmarks_data_np.shape[0] == 1:
                     df_landmarks = pd.DataFrame(landmarks_data_np, columns=landmark_cols)
                else:
                    logger.error(f"Unexpected landmarks_data_np shape for feature extraction: {landmarks_data_np.shape}")
                    return self.create_fallback_features(landmarks_data_np, language)


                # Call the external feature extractor
                # Ensure extract_features function can handle language as an argument if needed
                # If extract_features only takes DataFrame, remove language from call
                features_df = extract_features(df_landmarks, perform_selection=False) # Or True if used in training
                
                if not features_df.empty:
                    # Drop target columns if they exist, to prepare for prediction
                    features_for_prediction = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
                    
                    logger.info(f"Features extracted using pipeline: {features_for_prediction.shape}")
                    return features_for_prediction
            
            logger.warning("Feature extractor pipeline not available or landmarks data is None. Using fallback features.")
            return self.create_fallback_features(landmarks_data_np, language)
            
        except Exception as e:
            logger.error(f"Feature extraction using pipeline failed: {e}")
            logger.warning("Using fallback features due to pipeline failure.")
            return self.create_fallback_features(landmarks_data_np, language)
    
    def create_fallback_features(self, landmarks_data, language):
        """Create fallback features (if feature_extractor fails or is unavailable)"""
        try:
            features = {}
            
            # Ensure landmarks_data is a numpy array for slicing
            if isinstance(landmarks_data, list):
                landmarks_data = np.array(landmarks_data, dtype=np.float32)

            # Split into two hands
            hand1_data = landmarks_data[:63] if landmarks_data is not None and len(landmarks_data) >= 63 else np.zeros(63)
            hand2_data = landmarks_data[63:] if landmarks_data is not None and len(landmarks_data) == 126 else np.zeros(63)
            
            feature_idx = 0
            
            for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
                # Check if hand data is not just zeros (means hand was detected)
                hand_exists = not np.all(hand_data == 0.0) # More robust check for all zeros

                if hand_exists and len(hand_data) >= 63: # Ensure enough data for landmarks
                    x_coords = hand_data[0::3]
                    y_coords = hand_data[1::3]
                    z_coords = hand_data[2::3]
                    
                    # Basic statistics
                    stats = [
                        np.mean(x_coords) if x_coords.size > 0 else 0.0, np.std(x_coords) if x_coords.size > 0 else 0.0, np.min(x_coords) if x_coords.size > 0 else 0.0, np.max(x_coords) if x_coords.size > 0 else 0.0,
                        np.mean(y_coords) if y_coords.size > 0 else 0.0, np.std(y_coords) if y_coords.size > 0 else 0.0, np.min(y_coords) if y_coords.size > 0 else 0.0, np.max(y_coords) if y_coords.size > 0 else 0.0,
                        np.mean(z_coords) if z_coords.size > 0 else 0.0, np.std(z_coords) if z_coords.size > 0 else 0.0, np.min(z_coords) if z_coords.size > 0 else 0.0, np.max(z_coords) if z_coords.size > 0 else 0.0,
                    ]
                    
                    for stat in stats:
                        if feature_idx < 80: # Ensure we don't exceed max feature count
                            features[f'feature_{feature_idx}'] = float(stat) if np.isfinite(stat) else 0.0
                            feature_idx += 1
                    
                    # Distances from wrist (landmark 0)
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    # Finger tip landmarks (4, 8, 12, 16, 20)
                    for tip_idx_in_hand in [4, 8, 12, 16, 20]:
                        if tip_idx_in_hand < len(x_coords) and feature_idx < 80:
                            dist = np.sqrt((x_coords[tip_idx_in_hand] - wrist_x)**2 + 
                                           (y_coords[tip_idx_in_hand] - wrist_y)**2 + 
                                           (z_coords[tip_idx_in_hand] - wrist_z)**2)
                            features[f'feature_{feature_idx}'] = float(dist) if np.isfinite(dist) else 0.0
                            feature_idx += 1
                    
                    # Hand geometry (width and height from bounding box)
                    if feature_idx < 80:
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        features[f'feature_{feature_idx}'] = float(width) if np.isfinite(width) else 0.0
                        feature_idx += 1
                    if feature_idx < 80:
                        features[f'feature_{feature_idx}'] = float(height) if np.isfinite(height) else 0.0
                        feature_idx += 1
                else:
                    # Fill with zeros for missing or invalid hand data
                    # Each hand contributes approx 12 (stats) + 5 (distances) + 2 (geometry) = 19 features
                    # To match potential sklearn model input size (e.g., 50 or more features)
                    num_expected_hand_features = 19 # Example number of features per hand
                    for _ in range(num_expected_hand_features):
                        if feature_idx < 80: # Ensure we don't exceed max feature count
                            features[f'feature_{feature_idx}'] = 0.0
                            feature_idx += 1
            
            # Two-hand interaction features for BISINDO (if both hands detected)
            if np.all(hand1_data != 0.0) and np.all(hand2_data != 0.0) and feature_idx < 80:
                wrist1_x, wrist1_y = hand1_data[0], hand1_data[1]
                wrist2_x, wrist2_y = hand2_data[0], hand2_data[1]
                
                # Distance between wrists
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
            else:
                # Pad with zeros for missing two-hand features if hands not detected
                num_expected_two_hand_features = 3 # Example number of two-hand features
                for _ in range(num_expected_two_hand_features):
                    if feature_idx < 80:
                        features[f'feature_{feature_idx}'] = 0.0
                        feature_idx += 1

            # Pad to the minimum expected feature count for models (e.g., 50, 80, or 126)
            # This is crucial for models expecting a fixed number of input features.
            # You should know the exact number of features your model expects.
            # If your model expects 80 features, use 80 here.
            # If you are using feature_names from model data, this padding is less critical.
            # Example: Pad to 80 features, or more if needed by your model.
            min_expected_features = 80 # Adjust this based on your model's input feature count
            while feature_idx < min_expected_features:
                features[f'feature_{feature_idx}'] = 0.0
                feature_idx += 1

            logger.info(f"Fallback features created: {len(features)} values.")
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Fallback feature creation failed: {e}")
            # Ensure a DataFrame is always returned, even if empty or filled with zeros.
            basic_features = {f'feature_{i}': 0.0 for i in range(80)} # Create a default DataFrame
            return pd.DataFrame([basic_features])
    
    def predict_with_model(self, features_df, language):
        """Make prediction using available model (sklearn preferred, then TensorFlow)"""
        try:
            model_info = self.models.get(language.upper()) # Ensure language matches model keys (e.g., 'BISINDO')
            if not model_info:
                logger.warning(f"No model info found for language {language.upper()}")
                return None, 0.0
            
            # Prefer sklearn model for stability
            if 'sklearn' in model_info['available_models']:
                logger.info(f"Using sklearn model for {language.upper()}")
                return self.predict_sklearn(features_df, model_info['sklearn_model'])
            elif 'tensorflow' in model_info['available_models']:
                logger.info(f"Using TensorFlow model for {language.upper()}")
                return self.predict_tensorflow(features_df, model_info['tensorflow_model'], model_info['tensorflow_meta'])
            else:
                logger.warning(f"No usable model found for {language.upper()}")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Prediction dispatch failed for {language}: {e}")
            return None, 0.0
    
    def predict_sklearn(self, features_df, model_data):
        """Predict using sklearn model"""
        try:
            model = model_data.get('model')
            scaler = model_data.get('scaler') # StandardScaler or MinMaxScaler
            feature_names = model_data.get('feature_names') # List of feature names as expected by the model
            
            if model is None:
                raise ValueError("Sklearn model object is missing.")

            # Ensure features_df has the correct columns and order for prediction
            if feature_names:
                # Add missing features as 0.0 and drop extra ones
                current_cols = set(features_df.columns)
                expected_cols = set(feature_names)

                # Add missing columns
                for feature in expected_cols - current_cols:
                    features_df[feature] = 0.0
                # Drop extra columns
                for feature in current_cols - expected_cols:
                    features_df = features_df.drop(columns=[feature])

                features_df = features_df[feature_names] # Ensure order is correct
                logger.info(f"Sklearn features aligned to expected {len(feature_names)} features.")
            else:
                logger.warning("Sklearn model metadata does not contain feature_names. Proceeding with existing features_df columns. This might lead to incorrect predictions.")

            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
                logger.info("Features scaled using provided scaler.")
            else:
                features_scaled = features_df.values
                logger.warning("No scaler provided for sklearn model. Features not scaled.")
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Get confidence
            confidence = 0.0
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = np.max(probabilities)
                elif hasattr(model, 'decision_function'): # For SVMs
                    decision_values = model.decision_function(features_scaled)[0]
                    # Convert decision values to probabilities (approximate)
                    confidence = np.max(np.exp(decision_values) / np.sum(np.exp(decision_values)))
                else:
                    confidence = 0.7 # Default confidence if no proba/decision_function
                logger.info(f"Sklearn prediction confidence: {confidence:.3f}")
            except Exception as conf_e:
                logger.warning(f"Could not get sklearn prediction confidence: {conf_e}. Defaulting to 0.7")
                confidence = 0.7
            
            return str(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Sklearn prediction error: {e}")
            return None, 0.0
    
    def predict_tensorflow(self, features_df, model, meta):
        """Predict using TensorFlow model"""
        try:
            scaler = meta.get('scaler')
            label_encoder = meta.get('label_encoder')
            feature_names = meta.get('feature_names')
            
            if model is None:
                raise ValueError("TensorFlow model object is missing.")

            # Handle feature compatibility
            if feature_names:
                current_cols = set(features_df.columns)
                expected_cols = set(feature_names)
                for feature in expected_cols - current_cols:
                    features_df[feature] = 0.0
                for feature in current_cols - expected_cols:
                    features_df = features_df.drop(columns=[feature])
                features_df = features_df[feature_names]
                logger.info(f"TensorFlow features aligned to expected {len(feature_names)} features.")
            else:
                logger.warning("TensorFlow model metadata does not contain feature_names. Proceeding with existing features_df columns. This might lead to incorrect predictions.")
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
                logger.info("Features scaled using provided scaler.")
            else:
                features_scaled = features_df.values
                logger.warning("No scaler provided for TensorFlow model. Features not scaled.")
            
            # Ensure input shape matches TensorFlow model's expectation
            # TensorFlow models often expect (batch_size, input_features) for dense layers
            # or (batch_size, height, width, channels) for CNNs
            if model.input_shape[1] != features_scaled.shape[1]:
                logger.error(f"TensorFlow model input feature count mismatch! Expected {model.input_shape[1]}, got {features_scaled.shape[1]}. Padding/Truncating.")
                # Attempt to pad/truncate if mismatch
                target_features_count = model.input_shape[1]
                if features_scaled.shape[1] < target_features_count:
                    padding = np.zeros((features_scaled.shape[0], target_features_count - features_scaled.shape[1]), dtype=features_scaled.dtype)
                    features_scaled = np.hstack((features_scaled, padding))
                elif features_scaled.shape[1] > target_features_count:
                    features_scaled = features_scaled[:, :target_features_count]
                
            predictions_prob = model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(predictions_prob, axis=1)[0]
            confidence = np.max(predictions_prob)
            
            # Convert to label
            if label_encoder:
                prediction = label_encoder.inverse_transform([predicted_class_idx])[0]
                logger.info(f"TensorFlow prediction: {prediction} (idx: {predicted_class_idx})")
            else:
                prediction = f"Class_{predicted_class_idx}"
                logger.warning("No label_encoder provided for TensorFlow model.")
            
            return str(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"TensorFlow prediction error: {e}")
            return None, 0.0
    
    def predict_sign(self, image_bgr, language_type='bisindo', mirror_mode=None):
        """Main prediction function with enhanced two-hand support"""
        try:
            language_type_upper = language_type.upper() # Standardize language type to upper
            logger.info(f"Predicting for {language_type_upper} (mirror_mode={mirror_mode})")
            
            if language_type_upper not in self.models:
                available = list(self.models.keys())
                error_msg = f"Model not available for {language_type_upper}. Available: {available}"
                logger.error(error_msg)
                # Fallback to general demo if language specific model is not found
                return self.predict_demo_fallback(image_bgr, language_type_upper), 0.7, error_msg 
            
            # Extract landmarks with enhanced two-hand detection
            landmarks_np = self.extract_landmarks_from_frame(image_bgr, mirror_mode=mirror_mode)
            if landmarks_np is None: # Catches error during extraction
                logger.warning("Landmark extraction totally failed. Returning 'No hand detected'.")
                return "No hand detected", 0.0, "Landmark extraction failed"
            
            if np.all(landmarks_np == 0.0): # Check if all landmarks are zeros (no hands detected)
                logger.warning("No hand landmarks detected after extraction. Returning 'No hand detected'.")
                return "No hand detected", 0.0, "No hand landmarks detected"
            
            # Extract features using pipeline (or fallback if pipeline fails/unavailable)
            features_df = self.extract_features_using_pipeline(landmarks_np, language_type_upper)
            
            if features_df.empty:
                logger.error("Feature extraction failed, resulting in empty DataFrame.")
                return "Feature error", 0.0, "Feature extraction failed" # Indicate feature extraction issue
            
            # Make prediction
            prediction, confidence = self.predict_with_model(features_df, language_type_upper)
            
            if prediction is None:
                logger.error("Model prediction failed. Falling back to demo.")
                # Fallback to intelligent demo if model prediction fails
                return self.predict_demo_fallback(image_bgr, language_type_upper), 0.7, "Model prediction failed, using demo"
                
            # Check for two-hand detection status (for informational purposes)
            hand1_exists = not np.all(landmarks_np[0:63] == 0.0)
            hand2_exists = not np.all(landmarks_np[63:126] == 0.0)
            hands_detected_count = sum([hand1_exists, hand2_exists])
            
            mirror_info = f" (mirror={mirror_mode})" if mirror_mode is not None else ""
            logger.info(f"SUCCESS{mirror_info}: {prediction} (confidence: {confidence:.3f}, hands: {hands_detected_count})")
            
            return prediction, confidence, f"Success - {hands_detected_count} hand(s) detected"
            
        except Exception as e:
            logger.error(f"Prediction function overall error: {e}")
            return self.predict_demo_fallback(image_bgr, language_type), 0.7, f"Prediction error: {str(e)}"

    def predict_demo_fallback(self, image_bgr, language_type):
        """Fallback to demo prediction if real prediction fails"""
        import random
        import hashlib
        
        try:
            # Generate a pseudo-deterministic prediction based on image hash
            # Convert BGR image to bytes for hashing
            is_success, im_buf_arr = cv2.imencode(".jpg", image_bgr)
            if is_success:
                byte_im = im_buf_arr.tobytes()
                image_hash = hashlib.md5(byte_im).hexdigest()
            else:
                image_hash = str(random.random()) # Fallback if image cannot be encoded
            
            hash_int = int(image_hash[:8], 16)
            
            # ALPHABET ONLY based on original demo
            alphabet_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                               'U', 'V', 'W', 'X', 'Y', 'Z']
            
            char_index = hash_int % len(alphabet_labels)
            prediction = alphabet_labels[char_index]
            
            logger.info(f"🎲 Falling back to intelligent demo for {language_type}: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Critical error in demo fallback: {e}")
            return random.choice(['DEMO', 'SIGN']) # Absolute fallback
            

# Initialize API instance globally
api = EnhancedSignLanguageAPI()

# Ganti health check function di app.py dengan ini:

@app.route('/api/health', methods=['GET'])
def health_check():
    """Safe health check endpoint"""
    try:
        # Update health check to reflect detailed loading status
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
            'backend_version': '2.0.1-fixed',
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
def translate_sign_endpoint(): # Renamed to avoid confusion with class method
    try:
        logger.info("Enhanced translate endpoint called")
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        if not api.models:
            error_msg = 'No models loaded. Please ensure models are present and valid.'
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg, 'prediction': api.predict_demo_fallback(np.zeros((100,100,3), dtype=np.uint8), 'UNKNOWN')}), 500 # Pass dummy image for demo
            
        image_data_b64 = data['image']
        language_type = data.get('language_type', 'bisindo').lower()
        mirror_mode = data.get('mirror_mode', None) # Boolean or None

        # Decode Base64 to OpenCV image (BGR format)
        try:
            if ',' in image_data_b64:
                image_data_b64 = image_data_b64.split(',')[1] # Remove data URI prefix
            
            image_bytes = base64.b64decode(image_data_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Ensure it's BGR
            
            if image_bgr is None:
                raise ValueError("Could not decode image from base64. Check format.")
                
            if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
                raise ValueError("Decoded image has zero dimensions.")
            
            logger.info(f"Decoded image shape: {image_bgr.shape}")
                
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return jsonify({'success': False, 'error': f'Image decoding failed: {e}'}), 400
            
        # Make prediction with enhanced two-hand support
        prediction, confidence, message = api.predict_sign(image_bgr, language_type, mirror_mode=mirror_mode)
        
        response = {
            'success': prediction is not None and prediction != "No hand detected" and prediction != "Feature error",
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
        return jsonify({'success': False, 'error': str(e), 'prediction': api.predict_demo_fallback(np.zeros((100,100,3), dtype=np.uint8), 'UNKNOWN')}), 500 # Fallback demo prediction on error

@app.route('/api/models', methods=['GET'])
def get_models_endpoint(): # Renamed to avoid confusion with class method
    return jsonify({
        'available_models': list(api.models.keys()),
        'total_models': len(api.models),
        'hand_detection': 'enhanced_two_hand_support',
        'models_detail': {lang: {'available_types': info['available_models']} for lang, info in api.models.items()}
    })
    
@app.route("/models", methods=["GET"])
def list_models():
    if hasattr(api, 'models'):
        return jsonify({
            "status": "success",
            "models_loaded": list(api.models.keys())
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Model registry not found."
        }), 500

@app.route("/")
def index():
    return jsonify({
        "message": "Sign Language API is running ",
        "available_endpoints": [
            "/predict",
            "/models",
            "/health"
        ]
    })


# Main execution block
if __name__ == '__main__':
    print("\nENHANCED TWO-HAND SIGN LANGUAGE API")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Models loaded: {list(api.models.keys())}")
    print(f"Total models: {len(api.models)}")
    print(f"MediaPipe ready: {api.hands is not None}")
    print(f"Feature extractor available: {extract_features_available}")
    print("ENHANCEMENTS:")
    print("  ✓ Enhanced two-hand detection for BISINDO")
    print("  ✓ Better preprocessing and CLAHE")
    print("  ✓ Mirror-aware handedness processing")
    print("  ✓ Improved hand sorting algorithms")
    print("  ✓ Sklearn + TensorFlow model support")
    
    if not api.models:
        print("\nNO MODELS LOADED! API WILL USE DEMO/FALLBACK FOR PREDICTIONS.")
        print("Please ensure model files (.pkl, .h5, .joblib) are in 'data/models/' and feature_extractor.py is in 'src/data_preprocessing/'.")
    else:
        print("\nAPI READY FOR TWO-HAND PREDICTIONS!")
        for lang, info in api.models.items():
            available_types = ', '.join(info['available_models'])
            print(f"    {lang}: {available_types}")
    
    print("=" * 50)
    print("To test locally: python app.py")
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"💥 Server start failed: {e}")
        # Fallback to localhost if 0.0.0.0 fails (e.g. for local Windows dev)
        # However, for Railway, 0.0.0.0 is crucial.
        if HOST == '0.0.0.0': # Only try fallback if original host failed for some reason
            try:
                logger.warning("Attempting to run on 127.0.0.1 due to previous failure.")
                app.run(host='127.0.0.1', port=PORT, debug=False, threaded=True, use_reloader=False)
            except Exception as e2:
                logger.critical(f"💥 Failed to start server on both 0.0.0.0 and 127.0.0.1: {e2}")
                sys.exit(1)
        else:
            sys.exit(1)
