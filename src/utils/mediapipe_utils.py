# src/utils/mediapipe_utils.py

import cv2
import mediapipe as mp
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMediaPipe:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        
        # Different configurations for different use cases
        self.hands_training = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lower threshold for training
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        
        self.hands_inference = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
    def preprocess_for_two_hands(self, image):
        """Enhanced preprocessing specifically for two-hand detection"""
        try:
            # Resize to optimal resolution
            height, width = image.shape[:2]
            target_width = 1280
            target_height = 720
            
            if width != target_width or height != target_height:
                image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Enhance contrast for better hand separation
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to improve hand visibility
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            enhanced = cv2.merge((l_channel, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Slight blur to reduce noise while keeping hand edges
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def extract_landmarks_enhanced(self, image, is_mirrored=False, use_training_config=True):
        """Enhanced landmark extraction with focus on two-hand detection"""
        try:
            # Choose appropriate configuration
            hands_detector = self.hands_training if use_training_config else self.hands_inference
            
            # Preprocess image
            processed_image = self.preprocess_for_two_hands(image)
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Multiple attempts with different processing
            results = None
            attempts = [
                rgb_image,  # Original
                cv2.convertScaleAbs(rgb_image, alpha=1.2, beta=10),  # Brighter
                cv2.convertScaleAbs(rgb_image, alpha=0.8, beta=-10),  # Darker
            ]
            
            for attempt_idx, img_variant in enumerate(attempts):
                results = hands_detector.process(img_variant)
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                    logger.debug(f"Two hands detected on attempt {attempt_idx + 1}")
                    break
                elif results.multi_hand_landmarks:
                    logger.debug(f"One hand detected on attempt {attempt_idx + 1}")
                    # Continue trying for two hands
            
            if not results or not results.multi_hand_landmarks:
                logger.debug("No hands detected")
                return None
            
            # Process detected hands
            landmarks_flat = []
            hand_data = []
            
            # Collect hand information
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score
                wrist_x = hand_landmarks.landmark[0].x
                
                # Adjust handedness based on mirror state
                if is_mirrored:
                    adjusted_label = hand_label
                else:
                    adjusted_label = "Right" if hand_label == "Left" else "Left"
                
                hand_data.append({
                    'landmarks': hand_landmarks,
                    'original_label': hand_label,
                    'adjusted_label': adjusted_label,
                    'score': hand_score,
                    'wrist_x': wrist_x,
                    'index': i
                })
            
            # Enhanced sorting for consistent hand ordering
            # For BISINDO: prioritize having both hands with good confidence
            if len(hand_data) >= 2:
                # Sort by adjusted handedness (Left first, then Right)
                hand_data.sort(key=lambda x: (x['adjusted_label'] == 'Right', -x['score']))
                # Fixed: Use string formatting instead of nested f-strings
                hand_info = [(h['adjusted_label'], f"{h['score']:.2f}") for h in hand_data]
                logger.debug(f"Two hands detected: {hand_info}")
            else:
                # Single hand - just sort by confidence
                hand_data.sort(key=lambda x: -x['score'])
                logger.debug(f"Single hand detected: {hand_data[0]['adjusted_label']} ({hand_data[0]['score']:.2f})")
            
            # Extract landmarks for up to 2 hands
            for hand_idx in range(2):
                if hand_idx < len(hand_data):
                    hand_landmarks = hand_data[hand_idx]['landmarks']
                    
                    # Extract 21 landmarks * 3 coordinates = 63 values
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
            
            hands_detected = len(hand_data)
            logger.debug(f"Extracted landmarks: {len(landmarks_flat)} values, {hands_detected} hands")
            
            return landmarks_flat
            
        except Exception as e:
            logger.error(f"Enhanced landmark extraction error: {e}")
            return None
    
    def close(self):
        if hasattr(self, 'hands_training'):
            self.hands_training.close()
        if hasattr(self, 'hands_inference'):
            self.hands_inference.close()

# Global handler
_enhanced_handler = None

def get_enhanced_handler():
    global _enhanced_handler
    if _enhanced_handler is None:
        _enhanced_handler = EnhancedMediaPipe()
    return _enhanced_handler

def extract_hand_landmarks(image_path, augment_with_mirror=False):
    """Enhanced extraction with better two-hand detection"""
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return None if not augment_with_mirror else []
        
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None if not augment_with_mirror else []
        
        handler = get_enhanced_handler()
        
        if not augment_with_mirror:
            # Standard extraction
            landmarks = handler.extract_landmarks_enhanced(image, is_mirrored=False, use_training_config=True)
            return landmarks
        else:
            # Augmented extraction for training
            results = []
            
            # Process original
            landmarks_original = handler.extract_landmarks_enhanced(image, is_mirrored=False, use_training_config=True)
            if landmarks_original is not None:
                results.append((landmarks_original, False))
            
            # Process mirrored
            mirrored_image = cv2.flip(image, 1)
            landmarks_mirrored = handler.extract_landmarks_enhanced(mirrored_image, is_mirrored=True, use_training_config=True)
            if landmarks_mirrored is not None:
                results.append((landmarks_mirrored, True))
            
            return results
            
    except Exception as e:
        logger.error(f"Error in extract_hand_landmarks: {e}")
        return None if not augment_with_mirror else []

def extract_hand_landmarks_for_inference(image_array, is_mirrored=False):
    """Extract landmarks for real-time inference with enhanced two-hand detection"""
    try:
        handler = get_enhanced_handler()
        return handler.extract_landmarks_enhanced(image_array, is_mirrored=is_mirrored, use_training_config=False)
    except Exception as e:
        logger.error(f"Error in inference extraction: {e}")
        return None

def validate_landmarks(landmarks_array):
    """Enhanced validation for two-hand landmarks"""
    if landmarks_array is None:
        return False
    
    if not isinstance(landmarks_array, np.ndarray):
        return False
    
    if landmarks_array.shape[0] != 126:
        logger.warning(f"Invalid landmarks shape: {landmarks_array.shape}, expected (126,)")
        return False
    
    # Check if at least one hand is detected (not all zeros)
    hand1_data = landmarks_array[:63]
    hand2_data = landmarks_array[63:]
    
    hand1_exists = not np.allclose(hand1_data[::3], 0) or not np.allclose(hand1_data[1::3], 0)
    hand2_exists = not np.allclose(hand2_data[::3], 0) or not np.allclose(hand2_data[1::3], 0)
    
    if not (hand1_exists or hand2_exists):
        logger.warning("No valid hand landmarks detected (all zeros)")
        return False
    
    # Check coordinate ranges
    x_coords = landmarks_array[0::3]
    y_coords = landmarks_array[1::3]
    
    # Allow some tolerance for edge cases
    if np.any(x_coords < -0.1) or np.any(x_coords > 1.1):
        logger.debug("X coordinates slightly out of range")
    
    if np.any(y_coords < -0.1) or np.any(y_coords > 1.1):
        logger.debug("Y coordinates slightly out of range")
    
    return True

def analyze_hand_detection_quality(image_path):
    """Analyze hand detection quality for debugging"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}
        
        handler = get_enhanced_handler()
        
        # Test with training config
        landmarks_training = handler.extract_landmarks_enhanced(image, use_training_config=True)
        
        # Test with inference config  
        landmarks_inference = handler.extract_landmarks_enhanced(image, use_training_config=False)
        
        # Analyze results
        analysis = {
            "image_path": image_path,
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "training_config": {
                "detected": landmarks_training is not None,
                "hands_count": 0
            },
            "inference_config": {
                "detected": landmarks_inference is not None,
                "hands_count": 0
            }
        }
        
        if landmarks_training is not None:
            hand1_exists = not np.allclose(landmarks_training[:63:3], 0)
            hand2_exists = not np.allclose(landmarks_training[63::3], 0)
            analysis["training_config"]["hands_count"] = int(hand1_exists) + int(hand2_exists)
        
        if landmarks_inference is not None:
            hand1_exists = not np.allclose(landmarks_inference[:63:3], 0)
            hand2_exists = not np.allclose(landmarks_inference[63::3], 0)
            analysis["inference_config"]["hands_count"] = int(hand1_exists) + int(hand2_exists)
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

def test_two_hand_detection(test_folder_path):
    """Test two-hand detection on a folder of images"""
    if not os.path.exists(test_folder_path):
        logger.error(f"Test folder not found: {test_folder_path}")
        return
    
    image_files = [f for f in os.listdir(test_folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        logger.error("No image files found in test folder")
        return
    
    logger.info(f"Testing two-hand detection on {len(image_files)} images...")
    
    results = {
        "total_images": len(image_files),
        "detected_any": 0,
        "detected_one_hand": 0,
        "detected_two_hands": 0,
        "failed": 0
    }
    
    for img_file in image_files[:10]:  # Test first 10 images
        img_path = os.path.join(test_folder_path, img_file)
        analysis = analyze_hand_detection_quality(img_path)
        
        if "error" in analysis:
            results["failed"] += 1
            continue
        
        hands_count = analysis["training_config"]["hands_count"]
        
        if hands_count > 0:
            results["detected_any"] += 1
            if hands_count == 1:
                results["detected_one_hand"] += 1
            elif hands_count == 2:
                results["detected_two_hands"] += 1
        else:
            results["failed"] += 1
    
    logger.info("Two-hand detection test results:")
    logger.info(f"  Total tested: {min(10, len(image_files))}")
    logger.info(f"  Any hands detected: {results['detected_any']}")
    logger.info(f"  One hand: {results['detected_one_hand']}")
    logger.info(f"  Two hands: {results['detected_two_hands']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Two-hand rate: {results['detected_two_hands']/min(10, len(image_files))*100:.1f}%")

# Cleanup
import atexit

def cleanup():
    global _enhanced_handler
    if _enhanced_handler is not None:
        _enhanced_handler.close()

atexit.register(cleanup)