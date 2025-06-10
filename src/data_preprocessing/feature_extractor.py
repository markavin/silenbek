# src/data_preprocessing/feature_extractor.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_basic_features(landmarks_data):
    """Create comprehensive basic features from landmarks"""
    try:
        features = {}
        
        # Split into 2 hands (21 landmarks * 3 coords = 63 values per hand)
        hand1_data = landmarks_data[:63]
        hand2_data = landmarks_data[63:]
        
        feature_idx = 0
        
        for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
            if len(hand_data) >= 63:
                x_coords = hand_data[::3]   # Every 3rd starting from 0 (x coordinates)
                y_coords = hand_data[1::3]  # Every 3rd starting from 1 (y coordinates) 
                z_coords = hand_data[2::3]  # Every 3rd starting from 2 (z coordinates)
                
                # Check if hand exists (not all zeros)
                hand_exists = not all(x == 0 and y == 0 for x, y in zip(x_coords, y_coords))
                
                if hand_exists:
                    # Basic statistics for each coordinate
                    stats = [
                        np.mean(x_coords), np.std(x_coords), np.min(x_coords), np.max(x_coords),
                        np.mean(y_coords), np.std(y_coords), np.min(y_coords), np.max(y_coords),
                        np.mean(z_coords), np.std(z_coords), np.min(z_coords), np.max(z_coords),
                    ]
                    
                    for i, stat in enumerate(stats):
                        features[f'h{hand_idx}_stat_{i}'] = float(stat) if np.isfinite(stat) else 0.0
                        feature_idx += 1
                    
                    # Wrist position (landmark 0)
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    
                    # Distances from wrist to fingertips
                    fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
                    for tip_idx in fingertip_indices:
                        if tip_idx < len(x_coords):
                            dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                         (y_coords[tip_idx] - wrist_y)**2 + 
                                         (z_coords[tip_idx] - wrist_z)**2)
                            features[f'h{hand_idx}_tip_{tip_idx}_dist'] = float(dist) if np.isfinite(dist) else 0.0
                            feature_idx += 1
                    
                    # Hand geometry
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    depth = max(z_coords) - min(z_coords)
                    
                    features[f'h{hand_idx}_width'] = float(width) if np.isfinite(width) else 0.0
                    features[f'h{hand_idx}_height'] = float(height) if np.isfinite(height) else 0.0
                    features[f'h{hand_idx}_depth'] = float(depth) if np.isfinite(depth) else 0.0
                    features[f'h{hand_idx}_aspect'] = float(width / (height + 1e-8)) if np.isfinite(width) and np.isfinite(height) else 0.0
                    feature_idx += 4
                    
                    # Inter-finger distances
                    for i in range(len(fingertip_indices)-1):
                        for j in range(i+1, len(fingertip_indices)):
                            tip1_idx = fingertip_indices[i]
                            tip2_idx = fingertip_indices[j]
                            
                            if tip1_idx < len(x_coords) and tip2_idx < len(x_coords):
                                dist = np.sqrt((x_coords[tip1_idx] - x_coords[tip2_idx])**2 + 
                                             (y_coords[tip1_idx] - y_coords[tip2_idx])**2 + 
                                             (z_coords[tip1_idx] - z_coords[tip2_idx])**2)
                                features[f'h{hand_idx}_inter_{tip1_idx}_{tip2_idx}'] = float(dist) if np.isfinite(dist) else 0.0
                                feature_idx += 1
                else:
                    # Fill with zeros for missing hand but keep feature names consistent
                    num_features_per_hand = 12 + 5 + 4 + 10  # stats + tip_dists + geometry + inter_dists
                    for i in range(num_features_per_hand):
                        features[f'h{hand_idx}_missing_{i}'] = 0.0
                        feature_idx += 1
            else:
                # Hand data incomplete
                num_features_per_hand = 12 + 5 + 4 + 10
                for i in range(num_features_per_hand):
                    features[f'h{hand_idx}_incomplete_{i}'] = 0.0
                    feature_idx += 1
        
        # Two-hand interaction features
        if len(hand1_data) >= 63 and len(hand2_data) >= 63:
            hand1_exists = not all(x == 0 and y == 0 for x, y in zip(hand1_data[::3], hand1_data[1::3]))
            hand2_exists = not all(x == 0 and y == 0 for x, y in zip(hand2_data[::3], hand2_data[1::3]))
            
            if hand1_exists and hand2_exists:
                # Distance between wrists
                wrist1_x, wrist1_y, wrist1_z = hand1_data[0], hand1_data[1], hand1_data[2]
                wrist2_x, wrist2_y, wrist2_z = hand2_data[0], hand2_data[1], hand2_data[2]
                
                inter_wrist_dist = np.sqrt((wrist1_x - wrist2_x)**2 + 
                                         (wrist1_y - wrist2_y)**2 + 
                                         (wrist1_z - wrist2_z)**2)
                features['inter_wrist_dist'] = float(inter_wrist_dist) if np.isfinite(inter_wrist_dist) else 0.0
                
                # Relative positions
                features['hands_relative_x'] = float(wrist1_x - wrist2_x) if np.isfinite(wrist1_x - wrist2_x) else 0.0
                features['hands_relative_y'] = float(wrist1_y - wrist2_y) if np.isfinite(wrist1_y - wrist2_y) else 0.0
                features['hands_relative_z'] = float(wrist1_z - wrist2_z) if np.isfinite(wrist1_z - wrist2_z) else 0.0
                
                # Cross-hand fingertip distances
                fingertip_indices = [4, 8, 12, 16, 20]
                for tip1_idx in fingertip_indices[:3]:  # Just first 3 to avoid too many features
                    for tip2_idx in fingertip_indices[:3]:
                        h1_tip_x = hand1_data[tip1_idx * 3]
                        h1_tip_y = hand1_data[tip1_idx * 3 + 1]
                        h1_tip_z = hand1_data[tip1_idx * 3 + 2]
                        
                        h2_tip_x = hand2_data[tip2_idx * 3]
                        h2_tip_y = hand2_data[tip2_idx * 3 + 1]
                        h2_tip_z = hand2_data[tip2_idx * 3 + 2]
                        
                        cross_dist = np.sqrt((h1_tip_x - h2_tip_x)**2 + 
                                           (h1_tip_y - h2_tip_y)**2 + 
                                           (h1_tip_z - h2_tip_z)**2)
                        features[f'cross_{tip1_idx}_{tip2_idx}'] = float(cross_dist) if np.isfinite(cross_dist) else 0.0
            else:
                # One or both hands missing
                features['inter_wrist_dist'] = 0.0
                features['hands_relative_x'] = 0.0
                features['hands_relative_y'] = 0.0
                features['hands_relative_z'] = 0.0
                
                # Fill cross distances with zeros
                fingertip_indices = [4, 8, 12, 16, 20]
                for tip1_idx in fingertip_indices[:3]:
                    for tip2_idx in fingertip_indices[:3]:
                        features[f'cross_{tip1_idx}_{tip2_idx}'] = 0.0
        
        # Add some raw landmark coordinates as features (most important ones)
        important_landmarks = [0, 4, 8, 12, 16, 20]  # wrist and fingertips
        for hand_idx in range(2):
            base_idx = hand_idx * 63
            for landmark_idx in important_landmarks:
                coord_idx = base_idx + landmark_idx * 3
                if coord_idx < len(landmarks_data):
                    features[f'h{hand_idx}_lm{landmark_idx}_x'] = float(landmarks_data[coord_idx])
                    features[f'h{hand_idx}_lm{landmark_idx}_y'] = float(landmarks_data[coord_idx + 1]) if coord_idx + 1 < len(landmarks_data) else 0.0
                    features[f'h{hand_idx}_lm{landmark_idx}_z'] = float(landmarks_data[coord_idx + 2]) if coord_idx + 2 < len(landmarks_data) else 0.0
                else:
                    features[f'h{hand_idx}_lm{landmark_idx}_x'] = 0.0
                    features[f'h{hand_idx}_lm{landmark_idx}_y'] = 0.0
                    features[f'h{hand_idx}_lm{landmark_idx}_z'] = 0.0
        
        logger.info(f"Created {len(features)} basic features")
        return pd.DataFrame([features])
        
    except Exception as e:
        logger.error(f"Basic feature creation failed: {e}")
        # Emergency fallback - simple coordinate features
        emergency_features = {}
        for i in range(min(126, len(landmarks_data))):
            emergency_features[f'coord_{i}'] = float(landmarks_data[i]) if np.isfinite(landmarks_data[i]) else 0.0
        
        # Pad to at least 50 features
        while len(emergency_features) < 50:
            emergency_features[f'pad_{len(emergency_features)}'] = 0.0
        
        logger.info(f"Using emergency fallback: {len(emergency_features)} features")
        return pd.DataFrame([emergency_features])

def extract_features(df_landmarks, max_features=80, perform_selection=True):
    """Main feature extraction function"""
    logger.info("Starting feature extraction...")
    
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
            
            # Create features for this sample
            features_df = create_basic_features(landmarks_data)
            
            if not features_df.empty:
                all_features.append(features_df.iloc[0].to_dict())
            else:
                logger.warning(f"Failed to extract features for row {idx}")
                # Create empty features
                empty_features = {f'empty_{i}': 0.0 for i in range(50)}
                all_features.append(empty_features)
        
        if not all_features:
            logger.error("No features extracted")
            return pd.DataFrame()
        
        # Create final DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Add metadata columns
        if 'label' in df_landmarks.columns:
            features_df['label'] = df_landmarks['label'].values
        if 'sign_language_type' in df_landmarks.columns:
            features_df['sign_language_type'] = df_landmarks['sign_language_type'].values
        if 'is_mirrored' in df_landmarks.columns:
            features_df['is_mirrored'] = df_landmarks['is_mirrored'].values
        
        # Clean features
        feature_columns = [col for col in features_df.columns 
                          if col not in ['label', 'sign_language_type', 'is_mirrored']]
        
        for col in feature_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)
            features_df[col] = features_df[col].replace([np.inf, -np.inf], 0.0)
        
        # Remove constant features
        feature_variances = features_df[feature_columns].var()
        constant_features = feature_variances[feature_variances == 0].index.tolist()
        
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            features_df = features_df.drop(columns=constant_features)
            feature_columns = [col for col in feature_columns if col not in constant_features]
        
        # Feature selection if needed and we have labels
        if perform_selection and 'label' in features_df.columns and len(feature_columns) > max_features:
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
                if 'label' in features_df.columns:
                    result_df['label'] = features_df['label']
                if 'sign_language_type' in features_df.columns:
                    result_df['sign_language_type'] = features_df['sign_language_type']
                if 'is_mirrored' in features_df.columns:
                    result_df['is_mirrored'] = features_df['is_mirrored']
                
                features_df = result_df
                
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}, using all features")
        
        final_feature_count = len([col for col in features_df.columns 
                                  if col not in ['label', 'sign_language_type', 'is_mirrored']])
        
        logger.info(f"Feature extraction completed: {final_feature_count} features, {len(features_df)} samples")
        
        if final_feature_count < 5:
            logger.error(f"Too few features: {final_feature_count}")
            return pd.DataFrame()
        
        return features_df
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return pd.DataFrame()

def analyze_feature_importance(features_df, top_k=10):
    """Analyze feature importance"""
    if 'label' not in features_df.columns:
        return None
    
    X = features_df.drop(columns=['label', 'sign_language_type', 'is_mirrored'], errors='ignore')
    y = features_df['label']
    
    try:
        scores = mutual_info_classif(X, y, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top {top_k} most important features:")
        for idx, row in feature_importance.head(top_k).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        return None