def create_standardized_features(landmarks_data):
    """Create standardized features with EXACT consistent naming for training/inference"""
    try:
        features = {}
        
        # Split into 2 hands (21 landmarks * 3 coords = 63 values per hand)
        hand1_data = landmarks_data[:63]
        hand2_data = landmarks_data[63:]
        
        # Process each hand with EXACT consistent feature naming
        for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
            hand_prefix = f'hand_{hand_idx}'
            
            if len(hand_data) >= 63:
                x_coords = hand_data[::3]   # x coordinates
                y_coords = hand_data[1::3]  # y coordinates
                z_coords = hand_data[2::3]  # z coordinates
                
                # Check if hand exists (not all zeros)
                hand_exists = not all(x == 0 and y == 0 and z == 0 for x, y, z in zip(x_coords, y_coords, z_coords))
                
                if hand_exists:
                    # SECTION 1: Basic statistics (12 features per hand) - EXACT naming
                    x_stats = [np.mean(x_coords), np.std(x_coords), np.min(x_coords), np.max(x_coords)]
                    y_stats = [np.mean(y_coords), np.std(y_coords), np.min(y_coords), np.max(y_coords)]
                    z_stats = [np.mean(z_coords), np.std(z_coords), np.min(z_coords), np.max(z_coords)]
                    
                    all_stats = x_stats + y_stats + z_stats
                    
                    for i, stat in enumerate(all_stats):
                        features[f'{hand_prefix}_stat_{i:02d}'] = float(stat) if np.isfinite(stat) else 0.0
                    
                    # SECTION 2: Fingertip distances from wrist (5 features per hand) - EXACT naming
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
                    
                    for i, tip_idx in enumerate(fingertip_indices):
                        if tip_idx < len(x_coords):
                            dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                         (y_coords[tip_idx] - wrist_y)**2 + 
                                         (z_coords[tip_idx] - wrist_z)**2)
                            features[f'{hand_prefix}_tip_dist_{i:02d}'] = float(dist) if np.isfinite(dist) else 0.0
                        else:
                            features[f'{hand_prefix}_tip_dist_{i:02d}'] = 0.0
                    
                    # SECTION 3: Hand geometry (4 features per hand) - EXACT naming
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    depth = max(z_coords) - min(z_coords)
                    aspect_ratio = width / (height + 1e-8)
                    
                    features[f'{hand_prefix}_width'] = float(width) if np.isfinite(width) else 0.0
                    features[f'{hand_prefix}_height'] = float(height) if np.isfinite(height) else 0.0
                    features[f'{hand_prefix}_depth'] = float(depth) if np.isfinite(depth) else 0.0
                    features[f'{hand_prefix}_aspect'] = float(aspect_ratio) if np.isfinite(aspect_ratio) else 0.0
                    
                    # SECTION 4: Inter-finger distances (10 features per hand) - EXACT naming
                    finger_distance_count = 0
                    for i in range(len(fingertip_indices)):
                        for j in range(i+1, len(fingertip_indices)):
                            tip1_idx = fingertip_indices[i]
                            tip2_idx = fingertip_indices[j]
                            
                            if tip1_idx < len(x_coords) and tip2_idx < len(x_coords):
                                dist = np.sqrt((x_coords[tip1_idx] - x_coords[tip2_idx])**2 + 
                                             (y_coords[tip1_idx] - y_coords[tip2_idx])**2 + 
                                             (z_coords[tip1_idx] - z_coords[tip2_idx])**2)
                                features[f'{hand_prefix}_inter_{finger_distance_count:02d}'] = float(dist) if np.isfinite(dist) else 0.0
                            else:
                                features[f'{hand_prefix}_inter_{finger_distance_count:02d}'] = 0.0
                            finger_distance_count += 1
                    
                    # SECTION 5: Important landmark coordinates (18 features per hand) - EXACT naming
                    important_landmarks = [0, 4, 8, 12, 16, 20]  # wrist and fingertips
                    for lm_idx in important_landmarks:
                        if lm_idx < len(x_coords):
                            features[f'{hand_prefix}_lm{lm_idx:02d}_x'] = float(x_coords[lm_idx])
                            features[f'{hand_prefix}_lm{lm_idx:02d}_y'] = float(y_coords[lm_idx])
                            features[f'{hand_prefix}_lm{lm_idx:02d}_z'] = float(z_coords[lm_idx])
                        else:
                            features[f'{hand_prefix}_lm{lm_idx:02d}_x'] = 0.0
                            features[f'{hand_prefix}_lm{lm_idx:02d}_y'] = 0.0
                            features[f'{hand_prefix}_lm{lm_idx:02d}_z'] = 0.0
                else:
                    # Hand doesn't exist - fill with zeros (49 features per hand) - EXACT naming
                    # 12 stats + 5 tip distances + 4 geometry + 10 inter-finger + 18 landmarks = 49
                    for i in range(12):
                        features[f'{hand_prefix}_stat_{i:02d}'] = 0.0
                    for i in range(5):
                        features[f'{hand_prefix}_tip_dist_{i:02d}'] = 0.0
                    features[f'{hand_prefix}_width'] = 0.0
                    features[f'{hand_prefix}_height'] = 0.0
                    features[f'{hand_prefix}_depth'] = 0.0
                    features[f'{hand_prefix}_aspect'] = 0.0
                    for i in range(10):
                        features[f'{hand_prefix}_inter_{i:02d}'] = 0.0
                    important_landmarks = [0, 4, 8, 12, 16, 20]
                    for lm_idx in important_landmarks:
                        features[f'{hand_prefix}_lm{lm_idx:02d}_x'] = 0.0
                        features[f'{hand_prefix}_lm{lm_idx:02d}_y'] = 0.0
                        features[f'{hand_prefix}_lm{lm_idx:02d}_z'] = 0.0
            else:
                # Hand data incomplete - fill with zeros (49 features per hand) - EXACT naming
                for i in range(12):
                    features[f'{hand_prefix}_stat_{i:02d}'] = 0.0
                for i in range(5):
                    features[f'{hand_prefix}_tip_dist_{i:02d}'] = 0.0
                features[f'{hand_prefix}_width'] = 0.0
                features[f'{hand_prefix}_height'] = 0.0
                features[f'{hand_prefix}_depth'] = 0.0
                features[f'{hand_prefix}_aspect'] = 0.0
                for i in range(10):
                    features[f'{hand_prefix}_inter_{i:02d}'] = 0.0
                important_landmarks = [0, 4, 8, 12, 16, 20]
                for lm_idx in important_landmarks:
                    features[f'{hand_prefix}_lm{lm_idx:02d}_x'] = 0.0
                    features[f'{hand_prefix}_lm{lm_idx:02d}_y'] = 0.0
                    features[f'{hand_prefix}_lm{lm_idx:02d}_z'] = 0.0
        
        # SECTION 6: Two-hand interaction features (13 features) - EXACT naming
        if len(hand1_data) >= 63 and len(hand2_data) >= 63:
            hand1_exists = not all(x == 0 and y == 0 and z == 0 for x, y, z in zip(hand1_data[::3], hand1_data[1::3], hand1_data[2::3]))
            hand2_exists = not all(x == 0 and y == 0 and z == 0 for x, y, z in zip(hand2_data[::3], hand2_data[1::3], hand2_data[2::3]))
            
            if hand1_exists and hand2_exists:
                # Wrist positions
                wrist1_x, wrist1_y, wrist1_z = hand1_data[0], hand1_data[1], hand1_data[2]
                wrist2_x, wrist2_y, wrist2_z = hand2_data[0], hand2_data[1], hand2_data[2]
                
                # Distance between wrists - EXACT naming
                inter_wrist_dist = np.sqrt((wrist1_x - wrist2_x)**2 + 
                                         (wrist1_y - wrist2_y)**2 + 
                                         (wrist1_z - wrist2_z)**2)
                features['inter_wrist_dist'] = float(inter_wrist_dist) if np.isfinite(inter_wrist_dist) else 0.0
                
                # Relative positions - EXACT naming
                features['hands_rel_x'] = float(wrist1_x - wrist2_x) if np.isfinite(wrist1_x - wrist2_x) else 0.0
                features['hands_rel_y'] = float(wrist1_y - wrist2_y) if np.isfinite(wrist1_y - wrist2_y) else 0.0
                features['hands_rel_z'] = float(wrist1_z - wrist2_z) if np.isfinite(wrist1_z - wrist2_z) else 0.0
                
                # Cross-hand fingertip distances (9 features) - EXACT naming
                cross_distances = []
                for tip1_idx in [4, 8, 12]:  # thumb, index, middle from hand1
                    for tip2_idx in [4, 8, 12]:  # thumb, index, middle from hand2
                        h1_tip_x = hand1_data[tip1_idx * 3]
                        h1_tip_y = hand1_data[tip1_idx * 3 + 1]
                        h1_tip_z = hand1_data[tip1_idx * 3 + 2]
                        
                        h2_tip_x = hand2_data[tip2_idx * 3]
                        h2_tip_y = hand2_data[tip2_idx * 3 + 1]
                        h2_tip_z = hand2_data[tip2_idx * 3 + 2]
                        
                        cross_dist = np.sqrt((h1_tip_x - h2_tip_x)**2 + 
                                           (h1_tip_y - h2_tip_y)**2 + 
                                           (h1_tip_z - h2_tip_z)**2)
                        cross_distances.append(float(cross_dist) if np.isfinite(cross_dist) else 0.0)
                
                # Add cross distances with EXACT naming
                for i, dist in enumerate(cross_distances):
                    features[f'cross_dist_{i:02d}'] = dist
            else:
                # One or both hands missing - fill interaction features with zeros - EXACT naming
                features['inter_wrist_dist'] = 0.0
                features['hands_rel_x'] = 0.0
                features['hands_rel_y'] = 0.0
                features['hands_rel_z'] = 0.0
                for i in range(9):
                    features[f'cross_dist_{i:02d}'] = 0.0
        else:
            # Insufficient hand data - fill interaction features with zeros - EXACT naming
            features['inter_wrist_dist'] = 0.0
            features['hands_rel_x'] = 0.0
            features['hands_rel_y'] = 0.0
            features['hands_rel_z'] = 0.0
            for i in range(9):
                features[f'cross_dist_{i:02d}'] = 0.0
        
        # CRITICAL: Sort features alphabetically for consistent order
        sorted_features = dict(sorted(features.items()))
        
        logger.debug(f"Created {len(sorted_features)} standardized features in alphabetical order")
        
        # Verify total feature count (should be 2*49 + 13 = 111)
        expected_total = 111
        if len(sorted_features) != expected_total:
            logger.warning(f"Feature count mismatch! Expected {expected_total}, got {len(sorted_features)}")
        
        # Log feature structure for debugging
        feature_types = {}
        for key in sorted_features.keys():
            if 'hand_0' in key:
                feature_types['hand_0'] = feature_types.get('hand_0', 0) + 1
            elif 'hand_1' in key:
                feature_types['hand_1'] = feature_types.get('hand_1', 0) + 1
            elif any(x in key for x in ['inter_', 'hands_', 'cross_']):
                feature_types['interaction'] = feature_types.get('interaction', 0) + 1
        
        logger.debug(f"Feature breakdown: {feature_types}")
        
        return pd.DataFrame([sorted_features])
        
    except Exception as e:
        logger.error(f"Standardized feature creation failed: {e}")
        # Emergency fallback with EXACT naming
        emergency_features = {}
        
        # Create exactly 111 features with consistent naming
        for hand_idx in range(2):
            hand_prefix = f'hand_{hand_idx}'
            for i in range(12):
                emergency_features[f'{hand_prefix}_stat_{i:02d}'] = 0.0
            for i in range(5):
                emergency_features[f'{hand_prefix}_tip_dist_{i:02d}'] = 0.0
            emergency_features[f'{hand_prefix}_width'] = 0.0
            emergency_features[f'{hand_prefix}_height'] = 0.0
            emergency_features[f'{hand_prefix}_depth'] = 0.0
            emergency_features[f'{hand_prefix}_aspect'] = 0.0
            for i in range(10):
                emergency_features[f'{hand_prefix}_inter_{i:02d}'] = 0.0
            for lm_idx in [0, 4, 8, 12, 16, 20]:
                emergency_features[f'{hand_prefix}_lm{lm_idx:02d}_x'] = 0.0
                emergency_features[f'{hand_prefix}_lm{lm_idx:02d}_y'] = 0.0
                emergency_features[f'{hand_prefix}_lm{lm_idx:02d}_z'] = 0.0
        
        # Interaction features
        emergency_features['inter_wrist_dist'] = 0.0
        emergency_features['hands_rel_x'] = 0.0
        emergency_features['hands_rel_y'] = 0.0
        emergency_features['hands_rel_z'] = 0.0
        for i in range(9):
            emergency_features[f'cross_dist_{i:02d}'] = 0.0
        
        # Sort emergency features too
        sorted_emergency = dict(sorted(emergency_features.items()))
        
        return pd.DataFrame([sorted_emergency])

def get_expected_feature_names():
    """Get the exact expected feature names in the correct order for model compatibility"""
    feature_names = []
    
    # Hand features (2 hands * 49 features each = 98)
    for hand_idx in range(2):
        hand_prefix = f'hand_{hand_idx}'
        
        # Stats (12 per hand)
        for i in range(12):
            feature_names.append(f'{hand_prefix}_stat_{i:02d}')
        
        # Tip distances (5 per hand)
        for i in range(5):
            feature_names.append(f'{hand_prefix}_tip_dist_{i:02d}')
        
        # Geometry (4 per hand)
        feature_names.extend([
            f'{hand_prefix}_width',
            f'{hand_prefix}_height', 
            f'{hand_prefix}_depth',
            f'{hand_prefix}_aspect'
        ])
        
        # Inter-finger distances (10 per hand)
        for i in range(10):
            feature_names.append(f'{hand_prefix}_inter_{i:02d}')
        
        # Landmark coordinates (18 per hand: 6 landmarks * 3 coords)
        for lm_idx in [0, 4, 8, 12, 16, 20]:
            feature_names.extend([
                f'{hand_prefix}_lm{lm_idx:02d}_x',
                f'{hand_prefix}_lm{lm_idx:02d}_y',
                f'{hand_prefix}_lm{lm_idx:02d}_z'
            ])
    
    # Interaction features (13 total)
    feature_names.extend([
        'inter_wrist_dist',
        'hands_rel_x',
        'hands_rel_y', 
        'hands_rel_z'
    ])
    
    # Cross distances (9)
    for i in range(9):
        feature_names.append(f'cross_dist_{i:02d}')
    
    # Sort alphabetically for consistency
    feature_names.sort()
    
    return feature_names

def validate_feature_names_consistency(features_df):
    """Validate that extracted features match expected names exactly"""
    expected_names = get_expected_feature_names()
    
    feature_columns = [col for col in features_df.columns 
                      if col not in ['label', 'sign_language_type', 'is_mirrored', 'image_name']]
    
    missing_features = set(expected_names) - set(feature_columns)
    extra_features = set(feature_columns) - set(expected_names)
    
    is_valid = True
    issues = []
    
    if missing_features:
        is_valid = False
        issues.append(f"Missing {len(missing_features)} expected features: {list(missing_features)[:5]}...")
    
    if extra_features:
        is_valid = False
        issues.append(f"Found {len(extra_features)} unexpected features: {list(extra_features)[:5]}...")
    
    if len(feature_columns) != len(expected_names):
        is_valid = False
        issues.append(f"Feature count mismatch: expected {len(expected_names)}, got {len(feature_columns)}")
    
    # Check order
    if feature_columns != expected_names:
        order_issues = []
        for i, (expected, actual) in enumerate(zip(expected_names[:10], feature_columns[:10])):
            if expected != actual:
                order_issues.append(f"pos {i}: expected '{expected}', got '{actual}'")
        if order_issues:
            issues.append(f"Feature order mismatch: {order_issues}")
    
    result = {
        'is_valid': is_valid,
        'issues': issues,
        'expected_count': len(expected_names),
        'actual_count': len(feature_columns),
        'match_ratio': len(set(expected_names) & set(feature_columns)) / len(expected_names)
    }
    
    if is_valid:
        logger.info("✅ Feature names validation PASSED - exact match with training format")
    else:
        logger.warning(f"❌ Feature names validation FAILED: {'; '.join(issues)}")
    
    return result# src/data_preprocessing/feature_extractor.py - FIXED VERSION

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_standardized_features(landmarks_data):
    """Create standardized features matching training format"""
    try:
        features = {}
        
        # Split into 2 hands (21 landmarks * 3 coords = 63 values per hand)
        hand1_data = landmarks_data[:63]
        hand2_data = landmarks_data[63:]
        
        feature_idx = 0
        
        # Process each hand consistently
        for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
            hand_prefix = f'h{hand_idx}'
            
            if len(hand_data) >= 63:
                x_coords = hand_data[::3]   # x coordinates
                y_coords = hand_data[1::3]  # y coordinates
                z_coords = hand_data[2::3]  # z coordinates
                
                # Check if hand exists (not all zeros)
                hand_exists = not all(x == 0 and y == 0 and z == 0 for x, y, z in zip(x_coords, y_coords, z_coords))
                
                if hand_exists:
                    # SECTION 1: Basic statistics (12 features per hand)
                    x_stats = [np.mean(x_coords), np.std(x_coords), np.min(x_coords), np.max(x_coords)]
                    y_stats = [np.mean(y_coords), np.std(y_coords), np.min(y_coords), np.max(y_coords)]
                    z_stats = [np.mean(z_coords), np.std(z_coords), np.min(z_coords), np.max(z_coords)]
                    
                    all_stats = x_stats + y_stats + z_stats
                    
                    for i, stat in enumerate(all_stats):
                        features[f'{hand_prefix}_stat_{i}'] = float(stat) if np.isfinite(stat) else 0.0
                    
                    # SECTION 2: Fingertip distances from wrist (5 features per hand)
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
                    
                    for i, tip_idx in enumerate(fingertip_indices):
                        if tip_idx < len(x_coords):
                            dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                         (y_coords[tip_idx] - wrist_y)**2 + 
                                         (z_coords[tip_idx] - wrist_z)**2)
                            features[f'{hand_prefix}_tip_dist_{i}'] = float(dist) if np.isfinite(dist) else 0.0
                        else:
                            features[f'{hand_prefix}_tip_dist_{i}'] = 0.0
                    
                    # SECTION 3: Hand geometry (4 features per hand)
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    depth = max(z_coords) - min(z_coords)
                    aspect_ratio = width / (height + 1e-8)
                    
                    features[f'{hand_prefix}_width'] = float(width) if np.isfinite(width) else 0.0
                    features[f'{hand_prefix}_height'] = float(height) if np.isfinite(height) else 0.0
                    features[f'{hand_prefix}_depth'] = float(depth) if np.isfinite(depth) else 0.0
                    features[f'{hand_prefix}_aspect'] = float(aspect_ratio) if np.isfinite(aspect_ratio) else 0.0
                    
                    # SECTION 4: Inter-finger distances (10 features per hand)
                    finger_distance_count = 0
                    for i in range(len(fingertip_indices)):
                        for j in range(i+1, len(fingertip_indices)):
                            tip1_idx = fingertip_indices[i]
                            tip2_idx = fingertip_indices[j]
                            
                            if tip1_idx < len(x_coords) and tip2_idx < len(x_coords):
                                dist = np.sqrt((x_coords[tip1_idx] - x_coords[tip2_idx])**2 + 
                                             (y_coords[tip1_idx] - y_coords[tip2_idx])**2 + 
                                             (z_coords[tip1_idx] - z_coords[tip2_idx])**2)
                                features[f'{hand_prefix}_inter_{finger_distance_count}'] = float(dist) if np.isfinite(dist) else 0.0
                            else:
                                features[f'{hand_prefix}_inter_{finger_distance_count}'] = 0.0
                            finger_distance_count += 1
                    
                    # SECTION 5: Important landmark coordinates (18 features per hand)
                    important_landmarks = [0, 4, 8, 12, 16, 20]  # wrist and fingertips
                    for lm_idx in important_landmarks:
                        if lm_idx < len(x_coords):
                            features[f'{hand_prefix}_lm{lm_idx}_x'] = float(x_coords[lm_idx])
                            features[f'{hand_prefix}_lm{lm_idx}_y'] = float(y_coords[lm_idx])
                            features[f'{hand_prefix}_lm{lm_idx}_z'] = float(z_coords[lm_idx])
                        else:
                            features[f'{hand_prefix}_lm{lm_idx}_x'] = 0.0
                            features[f'{hand_prefix}_lm{lm_idx}_y'] = 0.0
                            features[f'{hand_prefix}_lm{lm_idx}_z'] = 0.0
                else:
                    # Hand doesn't exist - fill with zeros (49 features per hand)
                    # 12 stats + 5 tip distances + 4 geometry + 10 inter-finger + 18 landmarks
                    total_features_per_hand = 49
                    for i in range(total_features_per_hand):
                        features[f'{hand_prefix}_missing_{i}'] = 0.0
            else:
                # Hand data incomplete - fill with zeros
                total_features_per_hand = 49
                for i in range(total_features_per_hand):
                    features[f'{hand_prefix}_incomplete_{i}'] = 0.0
        
        # SECTION 6: Two-hand interaction features (12 features)
        if len(hand1_data) >= 63 and len(hand2_data) >= 63:
            hand1_exists = not all(x == 0 and y == 0 and z == 0 for x, y, z in zip(hand1_data[::3], hand1_data[1::3], hand1_data[2::3]))
            hand2_exists = not all(x == 0 and y == 0 and z == 0 for x, y, z in zip(hand2_data[::3], hand2_data[1::3], hand2_data[2::3]))
            
            if hand1_exists and hand2_exists:
                # Wrist positions
                wrist1_x, wrist1_y, wrist1_z = hand1_data[0], hand1_data[1], hand1_data[2]
                wrist2_x, wrist2_y, wrist2_z = hand2_data[0], hand2_data[1], hand2_data[2]
                
                # Distance between wrists
                inter_wrist_dist = np.sqrt((wrist1_x - wrist2_x)**2 + 
                                         (wrist1_y - wrist2_y)**2 + 
                                         (wrist1_z - wrist2_z)**2)
                features['inter_wrist_dist'] = float(inter_wrist_dist) if np.isfinite(inter_wrist_dist) else 0.0
                
                # Relative positions
                features['hands_rel_x'] = float(wrist1_x - wrist2_x) if np.isfinite(wrist1_x - wrist2_x) else 0.0
                features['hands_rel_y'] = float(wrist1_y - wrist2_y) if np.isfinite(wrist1_y - wrist2_y) else 0.0
                features['hands_rel_z'] = float(wrist1_z - wrist2_z) if np.isfinite(wrist1_z - wrist2_z) else 0.0
                
                # Cross-hand fingertip distances (top 3 fingers only to limit features)
                cross_distances = []
                for tip1_idx in [4, 8, 12]:  # thumb, index, middle from hand1
                    for tip2_idx in [4, 8, 12]:  # thumb, index, middle from hand2
                        h1_tip_x = hand1_data[tip1_idx * 3]
                        h1_tip_y = hand1_data[tip1_idx * 3 + 1]
                        h1_tip_z = hand1_data[tip1_idx * 3 + 2]
                        
                        h2_tip_x = hand2_data[tip2_idx * 3]
                        h2_tip_y = hand2_data[tip2_idx * 3 + 1]
                        h2_tip_z = hand2_data[tip2_idx * 3 + 2]
                        
                        cross_dist = np.sqrt((h1_tip_x - h2_tip_x)**2 + 
                                           (h1_tip_y - h2_tip_y)**2 + 
                                           (h1_tip_z - h2_tip_z)**2)
                        cross_distances.append(float(cross_dist) if np.isfinite(cross_dist) else 0.0)
                
                # Add cross distances (9 features)
                for i, dist in enumerate(cross_distances):
                    features[f'cross_dist_{i}'] = dist
            else:
                # One or both hands missing - fill interaction features with zeros
                features['inter_wrist_dist'] = 0.0
                features['hands_rel_x'] = 0.0
                features['hands_rel_y'] = 0.0
                features['hands_rel_z'] = 0.0
                for i in range(9):
                    features[f'cross_dist_{i}'] = 0.0
        else:
            # Insufficient hand data - fill interaction features with zeros
            features['inter_wrist_dist'] = 0.0
            features['hands_rel_x'] = 0.0
            features['hands_rel_y'] = 0.0
            features['hands_rel_z'] = 0.0
            for i in range(9):
                features[f'cross_dist_{i}'] = 0.0
        
        logger.debug(f"Created {len(features)} standardized features")
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.error(f"Standardized feature creation failed: {e}")
        # Emergency fallback - create minimal feature set
        emergency_features = {}
        
        # Create exactly 110 features (2*49 + 12 interaction)
        for i in range(110):
            emergency_features[f'feature_{i}'] = 0.0
        
        return pd.DataFrame([emergency_features])

def extract_features(df_landmarks, max_features=None, perform_selection=False):
    """Main feature extraction function with standardized output"""
    logger.info("Starting standardized feature extraction...")
    
    if df_landmarks.empty:
        logger.warning("Input DataFrame is empty")
        return pd.DataFrame()
    
    try:
        # Get landmark columns
        landmark_columns = [col for col in df_landmarks.columns if col.startswith('landmark_')]
        
        if len(landmark_columns) != 126:
            logger.warning(f"Expected 126 landmark columns, got {len(landmark_columns)}")
        
        # Extract features for each row
        all_features = []
        
        for idx, row in df_landmarks.iterrows():
            landmarks_data = row[landmark_columns].values.astype(float)
            
            # Replace any NaN with 0
            landmarks_data = np.where(np.isfinite(landmarks_data), landmarks_data, 0.0)
            
            # Create standardized features
            features_df = create_standardized_features(landmarks_data)
            
            if not features_df.empty:
                all_features.append(features_df.iloc[0].to_dict())
            else:
                logger.warning(f"Failed to extract features for row {idx}")
                # Create emergency features
                emergency_features = {f'feature_{i}': 0.0 for i in range(110)}
                all_features.append(emergency_features)
        
        if not all_features:
            logger.error("No features extracted")
            return pd.DataFrame()
        
        # Create final DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Add metadata columns if they exist
        metadata_columns = ['label', 'sign_language_type', 'is_mirrored', 'image_name']
        for col in metadata_columns:
            if col in df_landmarks.columns:
                features_df[col] = df_landmarks[col].values
        
        # Clean features
        feature_columns = [col for col in features_df.columns 
                          if col not in metadata_columns]
        
        # Ensure all features are numeric
        for col in feature_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
            features_df[col] = features_df[col].replace([np.inf, -np.inf], 0.0)
        
        # Remove constant features (variance = 0)
        if len(features_df) > 1:  # Only if we have multiple samples
            feature_variances = features_df[feature_columns].var()
            constant_features = feature_variances[feature_variances == 0].index.tolist()
            
            if constant_features:
                logger.info(f"Removing {len(constant_features)} constant features")
                features_df = features_df.drop(columns=constant_features)
                feature_columns = [col for col in feature_columns if col not in constant_features]
        
        # Feature selection (only if requested and we have labels)
        if perform_selection and 'label' in features_df.columns and max_features and len(feature_columns) > max_features:
            try:
                X = features_df[feature_columns]
                y = features_df['label']
                
                selector = SelectKBest(score_func=mutual_info_classif, k=min(max_features, len(feature_columns)))
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selector.get_support()[i]]
                
                logger.info(f"Selected {len(selected_features)} best features from {len(feature_columns)}")
                
                # Create result dataframe
                result_df = pd.DataFrame(X_selected, columns=selected_features, index=features_df.index)
                
                # Add metadata back
                for col in metadata_columns:
                    if col in features_df.columns:
                        result_df[col] = features_df[col]
                
                features_df = result_df
                
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}, using all features")
        
        final_feature_count = len([col for col in features_df.columns 
                                  if col not in metadata_columns])
        
        logger.info(f"Feature extraction completed: {final_feature_count} features, {len(features_df)} samples")
        
        if final_feature_count < 5:
            logger.error(f"Too few features: {final_feature_count}")
            return pd.DataFrame()
        
        return features_df
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return pd.DataFrame()

def validate_feature_consistency(features_df, expected_feature_names=None):
    """Validate that features match expected format"""
    if features_df.empty:
        return False, "Empty DataFrame"
    
    feature_columns = [col for col in features_df.columns 
                      if col not in ['label', 'sign_language_type', 'is_mirrored', 'image_name']]
    
    if expected_feature_names:
        missing_features = set(expected_feature_names) - set(feature_columns)
        extra_features = set(feature_columns) - set(expected_feature_names)
        
        if missing_features:
            logger.warning(f"Missing expected features: {list(missing_features)[:5]}...")
        if extra_features:
            logger.warning(f"Extra unexpected features: {list(extra_features)[:5]}...")
        
        # Check if at least 80% of expected features are present
        match_ratio = len(set(expected_feature_names) & set(feature_columns)) / len(expected_feature_names)
        if match_ratio < 0.8:
            return False, f"Feature mismatch: only {match_ratio:.1%} features match"
    
    # Check for NaN or infinite values
    numeric_issues = features_df[feature_columns].isin([np.inf, -np.inf]).sum().sum()
    nan_issues = features_df[feature_columns].isnull().sum().sum()
    
    if numeric_issues > 0:
        logger.warning(f"Found {numeric_issues} infinite values in features")
    if nan_issues > 0:
        logger.warning(f"Found {nan_issues} NaN values in features")
    
    return True, f"Validation passed: {len(feature_columns)} features"

def analyze_feature_importance(features_df, top_k=10):
    """Analyze feature importance using mutual information"""
    if 'label' not in features_df.columns:
        return None
    
    feature_columns = [col for col in features_df.columns 
                      if col not in ['label', 'sign_language_type', 'is_mirrored', 'image_name']]
    
    X = features_df[feature_columns]
    y = features_df['label']
    
    try:
        scores = mutual_info_classif(X, y, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top {top_k} most important features:")
        for idx, row in feature_importance.head(top_k).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        return None
