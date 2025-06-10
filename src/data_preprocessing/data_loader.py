# src/data_preprocessing/data_loader.py

import os
import pandas as pd
import logging
from src.utils.mediapipe_utils import extract_hand_landmarks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_dataset(dataset_root_path, sign_language_type, enable_mirror_augmentation=True):
    logger.info(f"Loading {sign_language_type} dataset with mirror-aware processing")
    logger.info(f"Dataset path: {dataset_root_path}")
    logger.info(f"Mirror augmentation: {'Enabled' if enable_mirror_augmentation else 'Disabled'}")
    
    if not os.path.exists(dataset_root_path):
        logger.error(f"Dataset path does not exist: {dataset_root_path}")
        return pd.DataFrame()
    
    if not os.path.isdir(dataset_root_path):
        logger.error(f"Dataset path is not a directory: {dataset_root_path}")
        return pd.DataFrame()
    
    try:
        class_folders = [f for f in os.listdir(dataset_root_path) 
                        if os.path.isdir(os.path.join(dataset_root_path, f))]
        class_folders.sort()
        
        if not class_folders:
            logger.error(f"No class folders found in {dataset_root_path}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(class_folders)} classes: {class_folders}")
        
    except Exception as e:
        logger.error(f"Error listing class folders: {e}")
        return pd.DataFrame()
    
    all_landmarks = []
    all_labels = []
    all_mirror_flags = []
    all_image_names = []
    
    total_images_processed = 0
    total_landmarks_extracted = 0
    failed_extractions = 0
    
    for class_idx, label_name in enumerate(class_folders):
        label_path = os.path.join(dataset_root_path, label_name)
        
        logger.info(f"Processing class {class_idx + 1}/{len(class_folders)}: {label_name}")
        
        try:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            image_files = [f for f in os.listdir(label_path) 
                          if f.lower().endswith(image_extensions)]
            
            if not image_files:
                logger.warning(f"No images found in {label_path}")
                continue
            
            logger.info(f"  Found {len(image_files)} images in {label_name}")
            
        except Exception as e:
            logger.error(f"Error listing images in {label_path}: {e}")
            continue
        
        class_landmarks_count = 0
        class_failed_count = 0
        
        for img_name in image_files:
            img_path = os.path.join(label_path, img_name)
            total_images_processed += 1
            
            try:
                if enable_mirror_augmentation:
                    augmented_results = extract_hand_landmarks(img_path, augment_with_mirror=True)
                    
                    for landmarks, is_mirrored in augmented_results:
                        if landmarks is not None:
                            all_landmarks.append(landmarks)
                            all_labels.append(label_name)
                            all_mirror_flags.append(is_mirrored)
                            
                            suffix = "_mirrored" if is_mirrored else "_original"
                            unique_name = f"{os.path.splitext(img_name)[0]}{suffix}"
                            all_image_names.append(unique_name)
                            
                            class_landmarks_count += 1
                            total_landmarks_extracted += 1
                        else:
                            class_failed_count += 1
                            failed_extractions += 1
                else:
                    landmarks = extract_hand_landmarks(img_path, augment_with_mirror=False)
                    
                    if landmarks is not None:
                        all_landmarks.append(landmarks)
                        all_labels.append(label_name)
                        all_mirror_flags.append(False)
                        all_image_names.append(img_name)
                        
                        class_landmarks_count += 1
                        total_landmarks_extracted += 1
                    else:
                        class_failed_count += 1
                        failed_extractions += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                class_failed_count += 1
                failed_extractions += 1
                continue
        
        logger.info(f"  {label_name}: {class_landmarks_count} landmarks extracted, {class_failed_count} failed")
    
    if not all_landmarks:
        logger.error(f"No landmarks extracted for {sign_language_type}")
        return pd.DataFrame()
    
    logger.info(f"Creating DataFrame from {len(all_landmarks)} landmark sets...")
    
    landmark_columns = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']]
    
    try:
        df = pd.DataFrame(all_landmarks, columns=landmark_columns)
        
        df['label'] = all_labels
        df['sign_language_type'] = sign_language_type
        df['is_mirrored'] = all_mirror_flags
        df['image_name'] = all_image_names
        
        logger.info(f"Dataset loading completed for {sign_language_type}:")
        logger.info(f"  Total images processed: {total_images_processed}")
        logger.info(f"  Total landmarks extracted: {total_landmarks_extracted}")
        logger.info(f"  Failed extractions: {failed_extractions}")
        logger.info(f"  Success rate: {(total_landmarks_extracted/(total_landmarks_extracted + failed_extractions)*100):.1f}%")
        logger.info(f"  Final DataFrame shape: {df.shape}")
        
        class_distribution = df['label'].value_counts().sort_index()
        logger.info(f"  Class distribution:")
        for label, count in class_distribution.items():
            mirror_count = df[(df['label'] == label) & (df['is_mirrored'] == True)].shape[0]
            original_count = count - mirror_count
            logger.info(f"    {label}: {count} total ({original_count} original + {mirror_count} mirrored)")
        
        if enable_mirror_augmentation:
            total_mirrored = df['is_mirrored'].sum()
            total_original = len(df) - total_mirrored
            logger.info(f"  Mirror distribution: {total_original} original + {total_mirrored} mirrored")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        return pd.DataFrame()

def load_dataset_for_training(dataset_root_path, sign_language_type):
    return load_and_process_dataset(
        dataset_root_path, 
        sign_language_type, 
        enable_mirror_augmentation=True
    )

def load_dataset_for_testing(dataset_root_path, sign_language_type):
    return load_and_process_dataset(
        dataset_root_path, 
        sign_language_type, 
        enable_mirror_augmentation=False
    )

def validate_dataset_quality(df, min_samples_per_class=3):
    if df.empty:
        return False, {"error": "Dataset is empty"}
    
    quality_report = {}
    
    quality_report['total_samples'] = len(df)
    quality_report['num_classes'] = df['label'].nunique()
    quality_report['has_mirror_data'] = 'is_mirrored' in df.columns
    
    class_counts = df['label'].value_counts()
    quality_report['class_distribution'] = class_counts.to_dict()
    quality_report['min_samples_per_class'] = class_counts.min()
    quality_report['max_samples_per_class'] = class_counts.max()
    quality_report['class_imbalance_ratio'] = class_counts.max() / class_counts.min()
    
    if 'is_mirrored' in df.columns:
        mirror_counts = df['is_mirrored'].value_counts()
        quality_report['mirror_distribution'] = {
            'original': mirror_counts.get(False, 0),
            'mirrored': mirror_counts.get(True, 0)
        }
    
    landmark_columns = [col for col in df.columns if col.startswith('landmark_')]
    
    missing_landmarks = df[landmark_columns].isnull().sum().sum()
    quality_report['missing_landmarks'] = missing_landmarks
    
    zero_only_rows = (df[landmark_columns] == 0).all(axis=1).sum()
    quality_report['zero_landmark_rows'] = zero_only_rows
    
    is_valid = True
    validation_issues = []
    
    if quality_report['min_samples_per_class'] < min_samples_per_class:
        is_valid = False
        validation_issues.append(f"Some classes have < {min_samples_per_class} samples")
    
    if quality_report['class_imbalance_ratio'] > 10:
        validation_issues.append("Severe class imbalance detected (ratio > 10:1)")
    
    if missing_landmarks > 0:
        validation_issues.append(f"{missing_landmarks} missing landmark values")
    
    if zero_only_rows > len(df) * 0.1:
        is_valid = False
        validation_issues.append(f"Too many failed landmark detections: {zero_only_rows}/{len(df)}")
    
    quality_report['validation_issues'] = validation_issues
    quality_report['is_valid'] = is_valid
    
    logger.info(f"Dataset Quality Report:")
    logger.info(f"  Total samples: {quality_report['total_samples']}")
    logger.info(f"  Classes: {quality_report['num_classes']}")
    logger.info(f"  Samples per class: {quality_report['min_samples_per_class']} - {quality_report['max_samples_per_class']}")
    logger.info(f"  Class imbalance ratio: {quality_report['class_imbalance_ratio']:.2f}")
    
    if quality_report['has_mirror_data']:
        logger.info(f"  Mirror data: {quality_report['mirror_distribution']['original']} original + {quality_report['mirror_distribution']['mirrored']} mirrored")
    
    if validation_issues:
        logger.warning(f"  Validation issues:")
        for issue in validation_issues:
            logger.warning(f"    - {issue}")
    else:
        logger.info(f"  Dataset quality validation passed")
    
    return is_valid, quality_report

def combine_datasets(*dataframes):
    if not dataframes:
        return pd.DataFrame()
    
    valid_dfs = [df for df in dataframes if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    try:
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        
        logger.info(f"Combined {len(valid_dfs)} datasets:")
        logger.info(f"  Total samples: {len(combined_df)}")
        logger.info(f"  Languages: {combined_df['sign_language_type'].unique()}")
        logger.info(f"  Classes: {combined_df['label'].nunique()}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        return pd.DataFrame()

def split_dataset_by_mirror(df):
    if 'is_mirrored' not in df.columns:
        logger.warning("No mirror information in dataset")
        return df, pd.DataFrame()
    
    original_df = df[df['is_mirrored'] == False].copy()
    mirrored_df = df[df['is_mirrored'] == True].copy()
    
    logger.info(f"Split dataset: {len(original_df)} original + {len(mirrored_df)} mirrored")
    
    return original_df, mirrored_df

def get_dataset_statistics(df):
    if df.empty:
        return {"error": "Dataset is empty"}
    
    stats = {}
    
    stats['total_samples'] = len(df)
    stats['total_features'] = len([col for col in df.columns if col.startswith('landmark_')])
    
    if 'label' in df.columns:
        class_counts = df['label'].value_counts()
        stats['num_classes'] = len(class_counts)
        stats['classes'] = list(class_counts.index)
        stats['class_distribution'] = class_counts.to_dict()
        stats['min_samples_per_class'] = class_counts.min()
        stats['max_samples_per_class'] = class_counts.max()
        stats['avg_samples_per_class'] = class_counts.mean()
    
    if 'sign_language_type' in df.columns:
        lang_counts = df['sign_language_type'].value_counts()
        stats['languages'] = list(lang_counts.index)
        stats['language_distribution'] = lang_counts.to_dict()
    
    if 'is_mirrored' in df.columns:
        mirror_counts = df['is_mirrored'].value_counts()
        stats['has_mirror_augmentation'] = True
        stats['original_samples'] = mirror_counts.get(False, 0)
        stats['mirrored_samples'] = mirror_counts.get(True, 0)
        stats['augmentation_ratio'] = stats['mirrored_samples'] / stats['original_samples'] if stats['original_samples'] > 0 else 0
    else:
        stats['has_mirror_augmentation'] = False
    
    landmark_columns = [col for col in df.columns if col.startswith('landmark_')]
    if landmark_columns:
        missing_values = df[landmark_columns].isnull().sum().sum()
        stats['missing_values'] = missing_values
        stats['missing_percentage'] = (missing_values / (len(df) * len(landmark_columns))) * 100
        
        zero_samples = (df[landmark_columns] == 0).all(axis=1).sum()
        stats['zero_samples'] = zero_samples
        stats['zero_percentage'] = (zero_samples / len(df)) * 100
        
        x_coords = df[[col for col in landmark_columns if '_x' in col]]
        y_coords = df[[col for col in landmark_columns if '_y' in col]]
        z_coords = df[[col for col in landmark_columns if '_z' in col]]
        
        stats['coordinate_ranges'] = {
            'x_range': [float(x_coords.min().min()), float(x_coords.max().max())],
            'y_range': [float(y_coords.min().min()), float(y_coords.max().max())],
            'z_range': [float(z_coords.min().min()), float(z_coords.max().max())]
        }
    
    return stats

def save_dataset_with_metadata(df, output_path, include_metadata=True):
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        
        if include_metadata:
            metadata_path = output_path.replace('.csv', '_metadata.json')
            stats = get_dataset_statistics(df)
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Metadata saved to {metadata_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return False

def load_dataset_from_csv(csv_path):
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded from {csv_path}: {df.shape}")
        
        required_columns = ['label']
        landmark_columns = [col for col in df.columns if col.startswith('landmark_')]
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return pd.DataFrame()
        
        if len(landmark_columns) != 126:
            logger.warning(f"Expected 126 landmark columns, found {len(landmark_columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset from CSV: {e}")
        return pd.DataFrame()

def load_and_process_dataset_legacy(dataset_root_path, sign_language_type):
    return load_and_process_dataset(
        dataset_root_path, 
        sign_language_type, 
        enable_mirror_augmentation=False
    )