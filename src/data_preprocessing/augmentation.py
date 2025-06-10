# src/data_preprocessing/augmentation.py

import cv2
import numpy as np
import os
import shutil
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def augment_image(image, rotation_range=(-10, 10), scale_range=(0.9, 1.1),
                  translation_range=(-0.05, 0.05), brightness_range=(0.8, 1.2)):
    """
    Menerapkan augmentasi pada sebuah gambar.
    
    Args:
        image (np.array): Gambar input (OpenCV format BGR).
        rotation_range (tuple): Rentang rotasi dalam derajat (min, max).
        scale_range (tuple): Rentang skala (min, max).
        translation_range (tuple): Rentang translasi sebagai fraksi dari lebar/tinggi gambar.
        brightness_range (tuple): Rentang perubahan brightness (min, max multiplier).
        
    Returns:
        np.array: Gambar yang telah diaugmentasi.
    """
    try:
        if image is None:
            return None
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotasi
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Skala
        scale = np.random.uniform(scale_range[0], scale_range[1])
        M_rot = M_rot * scale

        # Translasi
        tx = np.random.uniform(translation_range[0] * w, translation_range[1] * w)
        ty = np.random.uniform(translation_range[0] * h, translation_range[1] * h)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply transformasi
        augmented_image = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Ubah kecerahan
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        augmented_image = cv2.convertScaleAbs(augmented_image, alpha=brightness_factor, beta=0)
        
        return augmented_image
        
    except Exception as e:
        logger.error(f"Error in augment_image: {e}")
        return image  # Return original jika augmentation gagal

def perform_augmentation(input_dir, output_dir, num_augmentations_per_image=3):
    """
    Melakukan augmentasi pada gambar di input_dir dan menyimpannya di output_dir.
    
    Args:
        input_dir (str): Direktori yang berisi data mentah (e.g., 'data/raw/bisindo').
        output_dir (str): Direktori untuk menyimpan data yang telah diaugmentasi (e.g., 'data/augmented/bisindo').
        num_augmentations_per_image (int): Jumlah versi augmentasi untuk setiap gambar asli.
    """
    
    logger.info(f"Starting augmentation: {input_dir} -> {output_dir}")
    
    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return False
    
    if not os.path.isdir(input_dir):
        logger.error(f"Input path is not a directory: {input_dir}")
        return False
    
    # Create output directory
    try:
        if os.path.exists(output_dir):
            logger.info(f"Output directory exists, cleaning: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return False
    
    # Get list of class folders
    try:
        class_folders = [f for f in os.listdir(input_dir) 
                        if os.path.isdir(os.path.join(input_dir, f))]
        class_folders.sort()
        
        if not class_folders:
            logger.error(f"No class folders found in {input_dir}")
            return False
        
        logger.info(f"Found {len(class_folders)} class folders: {class_folders[:5]}{'...' if len(class_folders) > 5 else ''}")
        
    except Exception as e:
        logger.error(f"Error listing class folders: {e}")
        return False
    
    total_original_images = 0
    total_augmented_images = 0
    failed_images = 0
    
    start_time = time.time()
    
    # Process each class folder
    for class_idx, label_name in enumerate(class_folders):
        label_input_path = os.path.join(input_dir, label_name)
        label_output_path = os.path.join(output_dir, label_name)
        
        logger.info(f"Processing class {class_idx + 1}/{len(class_folders)}: {label_name}")
        
        try:
            # Create output class folder
            os.makedirs(label_output_path, exist_ok=True)
            
            # Get images in this class
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            images_in_folder = [f for f in os.listdir(label_input_path) 
                               if f.lower().endswith(image_extensions)]
            
            if not images_in_folder:
                logger.warning(f"No images found in {label_input_path}")
                continue
            
            logger.info(f"  Found {len(images_in_folder)} images in {label_name}")
            
            class_original_count = 0
            class_augmented_count = 0
            class_failed_count = 0
            
            # Process each image
            for img_idx, img_name in enumerate(images_in_folder):
                img_path = os.path.join(label_input_path, img_name)
                
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        logger.warning(f"Could not read image {img_path}")
                        class_failed_count += 1
                        continue
                    
                    # Copy original image to augmented folder
                    original_output_path = os.path.join(label_output_path, img_name)
                    cv2.imwrite(original_output_path, image)
                    class_original_count += 1
                    
                    # Generate augmented versions
                    base_name = os.path.splitext(img_name)[0]
                    extension = os.path.splitext(img_name)[1]
                    
                    for aug_idx in range(num_augmentations_per_image):
                        try:
                            augmented_img = augment_image(image)
                            
                            if augmented_img is not None:
                                aug_img_name = f"{base_name}_aug{aug_idx + 1}{extension}"
                                aug_output_path = os.path.join(label_output_path, aug_img_name)
                                cv2.imwrite(aug_output_path, augmented_img)
                                class_augmented_count += 1
                            else:
                                class_failed_count += 1
                                
                        except Exception as e:
                            logger.debug(f"Augmentation failed for {img_name} (aug {aug_idx + 1}): {e}")
                            class_failed_count += 1
                
                except Exception as e:
                    logger.warning(f"Error processing image {img_path}: {e}")
                    class_failed_count += 1
                    continue
            
            total_original_images += class_original_count
            total_augmented_images += class_augmented_count
            failed_images += class_failed_count
            
            logger.info(f"  {label_name}: {class_original_count} original + {class_augmented_count} augmented = {class_original_count + class_augmented_count} total")
            
            if class_failed_count > 0:
                logger.warning(f"  {label_name}: {class_failed_count} failed")
                
        except Exception as e:
            logger.error(f"Error processing class {label_name}: {e}")
            continue
    
    # Final summary
    processing_time = time.time() - start_time
    total_final_images = total_original_images + total_augmented_images
    
    logger.info(f"Augmentation completed:")
    logger.info(f"  Processing time: {processing_time/60:.1f} minutes")
    logger.info(f"  Original images: {total_original_images:,}")
    logger.info(f"  Augmented images: {total_augmented_images:,}")
    logger.info(f"  Total final images: {total_final_images:,}")
    logger.info(f"  Failed images: {failed_images:,}")
    logger.info(f"  Success rate: {(total_final_images / (total_final_images + failed_images) * 100):.1f}%")
    
    # Validate output
    if total_final_images == 0:
        logger.error("No images were successfully augmented")
        return False
    
    if total_final_images < total_original_images:
        logger.warning("Fewer final images than original - something went wrong")
    
    return True

def augment_dataset_for_language(language_type, project_root):
    """
    Augment dataset untuk bahasa tertentu
    
    Args:
        language_type (str): 'BISINDO' atau 'SIBI'
        project_root (str): Root directory project
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    
    logger.info(f"Starting augmentation for {language_type.upper()}")
    
    try:
        # Setup paths
        input_dir = os.path.join(project_root, 'data', 'raw', language_type.lower())
        output_dir = os.path.join(project_root, 'data', 'augmented', language_type.lower())
        
        # Validate input directory
        if not os.path.exists(input_dir):
            logger.error(f"Raw data directory not found: {input_dir}")
            return False
        
        # Perform augmentation
        success = perform_augmentation(input_dir, output_dir, num_augmentations_per_image=2)
        
        if success:
            logger.info(f"✓ {language_type.upper()} augmentation completed successfully")
        else:
            logger.error(f"✗ {language_type.upper()} augmentation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in augment_dataset_for_language for {language_type}: {e}")
        return False

def augment_all_datasets(project_root):
    """
    Augment semua datasets yang tersedia
    
    Args:
        project_root (str): Root directory project
        
    Returns:
        dict: Status augmentation untuk setiap bahasa
    """
    
    logger.info("Starting augmentation for all available datasets")
    
    languages = ['BISINDO', 'SIBI']
    results = {}
    
    for language in languages:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"AUGMENTING {language}")
            logger.info(f"{'='*50}")
            
            success = augment_dataset_for_language(language, project_root)
            results[language] = success
            
        except Exception as e:
            logger.error(f"Unexpected error augmenting {language}: {e}")
            results[language] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("AUGMENTATION SUMMARY")
    logger.info(f"{'='*50}")
    
    successful_languages = [lang for lang, success in results.items() if success]
    failed_languages = [lang for lang, success in results.items() if not success]
    
    logger.info(f"Successful: {len(successful_languages)} languages {successful_languages}")
    logger.info(f"Failed: {len(failed_languages)} languages {failed_languages}")
    
    return results