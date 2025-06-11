# src/models/train_model.py - FIXED COMPLETE VERSION

import pandas as pd
import numpy as np
import os
import pickle
import logging
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tensorflow_model = None
        self.sklearn_model = None
        self.feature_names = None
        
    def analyze_dataset(self, dataframe):
        """Analyze dataset characteristics"""
        logger.info("Analyzing dataset characteristics...")
        
        if dataframe.empty:
            logger.error("DataFrame is empty")
            return False, {}
        
        if 'label' not in dataframe.columns:
            logger.error("No 'label' column found")
            return False, {}
        
        label_counts = dataframe['label'].value_counts().sort_index()
        
        analysis = {
            'total_samples': len(dataframe),
            'num_classes': len(label_counts),
            'label_distribution': label_counts.to_dict(),
            'min_samples_per_class': label_counts.min(),
            'max_samples_per_class': label_counts.max(),
            'imbalance_ratio': label_counts.max() / label_counts.min()
        }
        
        logger.info(f"Dataset Analysis:")
        logger.info(f"  Total samples: {analysis['total_samples']}")
        logger.info(f"  Number of classes: {analysis['num_classes']}")
        logger.info(f"  Samples per class: {analysis['min_samples_per_class']} - {analysis['max_samples_per_class']}")
        logger.info(f"  Class imbalance ratio: {analysis['imbalance_ratio']:.2f}")
        
        if analysis['min_samples_per_class'] < 3:
            logger.error(f"Some classes have < 3 samples")
            return False, analysis
        
        return True, analysis
        
    def prepare_data(self, dataframe):
        """Prepare data with fixed preprocessing"""
        logger.info("Preparing data with fixed preprocessing...")
        
        is_valid, analysis = self.analyze_dataset(dataframe)
        if not is_valid:
            return None, None, None, None, None, None, None
        
        # Separate features and labels
        feature_cols = [col for col in dataframe.columns if col not in ['label', 'sign_language_type', 'is_mirrored']]
        X = dataframe[feature_cols].copy()
        y = dataframe['label'].copy()
        
        logger.info(f"Original data shape: X={X.shape}, y={y.shape}")
        
        # Data cleaning
        X = X.fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
        # More lenient variance threshold
        feature_variance = X.var()
        low_variance_threshold = 0.0001
        low_variance_features = feature_variance[feature_variance < low_variance_threshold].index
        
        if len(low_variance_features) > 0:
            logger.info(f"Removing {len(low_variance_features)} low-variance features (threshold: {low_variance_threshold})")
            X = X.drop(columns=low_variance_features)
        
        # If too few features, use all available
        if X.shape[1] < 5:
            logger.warning(f"Very few features remaining: {X.shape[1]}. Using all available features.")
            X = dataframe[feature_cols].copy()
            X = X.fillna(0.0)
            X = X.replace([np.inf, -np.inf], 0.0)
            
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            
            feature_variance = X.var()
            constant_features = feature_variance[feature_variance == 0].index
            
            if len(constant_features) > 0:
                logger.info(f"Removing {len(constant_features)} constant features")
                X = X.drop(columns=constant_features)
            
            if X.shape[1] == 0:
                logger.error("Too few features remaining: 0")
                return None, None, None, None, None, None, None
        
        logger.info(f"Features after filtering: {X.shape[1]}")
        
        # CRITICAL: Save feature names for model compatibility
        self.feature_names = X.columns.tolist()
        logger.info(f"CRITICAL: Saved {len(self.feature_names)} feature names for model compatibility")
        logger.info(f"First 10 feature names: {self.feature_names[:10]}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        y_categorical = to_categorical(y_encoded, num_classes)
        
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Train/test split
        test_size = 0.2 if len(X) > 100 else 0.15
        
        try:
            X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
                X, y, y_categorical, test_size=test_size, random_state=42, 
                stratify=y if analysis['min_samples_per_class'] >= 2 else None
            )
        except ValueError:
            logger.warning("Stratified split failed, using random split")
            X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
                X, y, y_categorical, test_size=test_size, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Final data shapes:")
        logger.info(f"  Train: {X_train_scaled.shape}")
        logger.info(f"  Test: {X_test_scaled.shape}")
        logger.info(f"  Features: {len(self.feature_names)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, y_train_cat, y_test_cat, analysis
    
    def build_adaptive_neural_network(self, input_shape, num_classes, dataset_size):
        """Build neural network adapted to dataset size and feature count"""
        logger.info(f"Building neural network for {num_classes} classes, {input_shape} features")
        
        # Adapt architecture to dataset size and feature count
        if dataset_size < 100 or input_shape < 20:
            architecture = [max(32, input_shape * 2), max(16, input_shape)]
            dropout_rate = 0.1
            l2_reg = 0.01
        elif dataset_size < 500 or input_shape < 50:
            architecture = [max(64, input_shape * 2), max(32, input_shape), 16]
            dropout_rate = 0.2
            l2_reg = 0.001
        else:
            architecture = [min(256, input_shape * 4), min(128, input_shape * 2), max(32, input_shape)]
            dropout_rate = 0.3
            l2_reg = 0.0001
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            architecture[0], 
            activation='relu', 
            input_shape=(input_shape,),
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in architecture[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate * 0.8))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax'))
        
        # Adaptive learning rate
        learning_rate = 0.001 if dataset_size > 200 else 0.01
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model architecture: {architecture}")
        logger.info(f"Learning rate: {learning_rate}, Dropout: {dropout_rate}")
        
        return model
    
    def train_tensorflow_model(self, X_train, y_train_cat, X_test, y_test_cat, analysis):
        """Train TensorFlow neural network"""
        logger.info("Training TensorFlow neural network...")
        
        num_classes = y_train_cat.shape[1]
        input_shape = X_train.shape[1]
        dataset_size = len(X_train)
        
        # Build model
        self.tensorflow_model = self.build_adaptive_neural_network(
            input_shape, num_classes, dataset_size
        )
        
        # Training parameters
        if dataset_size < 100:
            epochs = 200
            batch_size = min(16, dataset_size // 4)
            patience = 30
        elif dataset_size < 500:
            epochs = 150
            batch_size = 32
            patience = 25
        else:
            epochs = 100
            batch_size = 64
            patience = 20
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Handle class imbalance
        class_weight_dict = None
        if analysis['imbalance_ratio'] > 3:
            y_train_labels = np.argmax(y_train_cat, axis=1)
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train_labels), 
                y=y_train_labels
            )
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}
            logger.info(f"Using class weights for imbalanced data")
        
        # Train model
        logger.info(f"Training with epochs={epochs}, batch_size={batch_size}")
        
        history = self.tensorflow_model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        tf_loss, tf_accuracy = self.tensorflow_model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Get predictions
        y_pred_prob = self.tensorflow_model.predict(X_test, verbose=0)
        y_pred_tf = np.argmax(y_pred_prob, axis=1)
        y_test_labels = np.argmax(y_test_cat, axis=1)
        
        tf_f1 = f1_score(y_test_labels, y_pred_tf, average='macro')
        
        unique_predictions = len(np.unique(y_pred_tf))
        
        logger.info(f"TensorFlow Results:")
        logger.info(f"  Accuracy: {tf_accuracy:.4f}")
        logger.info(f"  F1-score: {tf_f1:.4f}")
        logger.info(f"  Prediction diversity: {unique_predictions}/{num_classes}")
        
        if unique_predictions < 2:
            logger.error("TensorFlow model shows poor prediction diversity")
            return None
        
        return {
            'model': self.tensorflow_model,
            'accuracy': tf_accuracy,
            'loss': tf_loss,
            'f1_score': tf_f1,
            'history': history,
            'predictions': y_pred_tf,
            'true_labels': y_test_labels,
            'unique_predictions': unique_predictions
        }
    
    def train_random_forest_model(self, X_train, y_train, X_test, y_test, analysis):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        dataset_size = len(X_train)
        num_features = X_train.shape[1]
        
        # Adaptive parameters
        if dataset_size < 100:
            n_estimators = 100
            max_depth = min(10, max(3, num_features // 2))
            min_samples_split = 2
            min_samples_leaf = 1
        elif dataset_size < 500:
            n_estimators = 150
            max_depth = min(15, max(5, num_features // 2))
            min_samples_split = 3
            min_samples_leaf = 2
        else:
            n_estimators = 200
            max_depth = min(20, max(8, num_features // 2))
            min_samples_split = 5
            min_samples_leaf = 2
        
        # Build model
        self.sklearn_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if analysis['imbalance_ratio'] > 2 else None
        )
        
        logger.info(f"Random Forest parameters:")
        logger.info(f"  n_estimators: {n_estimators}, max_depth: {max_depth}")
        
        # Cross-validation if enough data
        if len(X_train) > 30:
            cv_scores = cross_val_score(
                self.sklearn_model, X_train, y_train, 
                cv=min(5, analysis['min_samples_per_class']), 
                scoring='f1_macro'
            )
            logger.info(f"Cross-validation F1 scores: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        
        # Train model
        self.sklearn_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_sklearn = self.sklearn_model.predict(X_test)
        sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
        sklearn_f1 = f1_score(y_test, y_pred_sklearn, average='macro')
        
        unique_predictions = len(np.unique(y_pred_sklearn))
        num_classes = len(np.unique(y_train))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.sklearn_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Random Forest Results:")
        logger.info(f"  Accuracy: {sklearn_accuracy:.4f}")
        logger.info(f"  F1-score: {sklearn_f1:.4f}")
        logger.info(f"  Prediction diversity: {unique_predictions}/{num_classes}")
        
        logger.info("Top 5 important features:")
        for _, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        if unique_predictions < 2:
            logger.error("Random Forest model shows poor prediction diversity")
            return None
        
        return {
            'model': self.sklearn_model,
            'accuracy': sklearn_accuracy,
            'f1_score': sklearn_f1,
            'predictions': y_pred_sklearn,
            'true_labels': y_test,
            'unique_predictions': unique_predictions,
            'feature_importance': feature_importance
        }
    
    def select_best_model(self, tf_results, sklearn_results):
        """Select best performing model"""
        logger.info("Comparing model performances...")
        
        if tf_results is None and sklearn_results is None:
            logger.error("Both models failed to train")
            return None, 0
        
        if tf_results is None:
            logger.warning("TensorFlow model failed, using Random Forest")
            return 'sklearn', sklearn_results['f1_score']
        
        if sklearn_results is None:
            logger.warning("Random Forest model failed, using TensorFlow")
            return 'tensorflow', tf_results['f1_score']
        
        logger.info("Model Comparison:")
        logger.info(f"  TensorFlow - F1: {tf_results['f1_score']:.4f}, Acc: {tf_results['accuracy']:.4f}")
        logger.info(f"  Random Forest - F1: {sklearn_results['f1_score']:.4f}, Acc: {sklearn_results['accuracy']:.4f}")
        
        # Select based on F1 score
        if tf_results['f1_score'] > sklearn_results['f1_score']:
            best_model = 'tensorflow'
            best_performance = tf_results['f1_score']
        else:
            best_model = 'sklearn'
            best_performance = sklearn_results['f1_score']
        
        logger.info(f"Selected best model: {best_model.upper()} (F1: {best_performance:.4f})")
        
        return best_model, best_performance

def train_model(dataframe, model_save_base_name, language_type):
    """Main training function with FIXED feature name saving"""
    logger.info(f"Starting training for {language_type}")
    
    if dataframe.empty:
        logger.error(f"Input DataFrame for {language_type} is empty")
        return None, None
    
    try:
        trainer = FixedModelTrainer()
        
        result = trainer.prepare_data(dataframe)
        if result[0] is None:
            logger.error("Data preparation failed")
            return None, None
        
        X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, analysis = result
        
        # Train both models
        tf_results = trainer.train_tensorflow_model(
            X_train, y_train_cat, X_test, y_test_cat, analysis
        )
        
        sklearn_results = trainer.train_random_forest_model(
            X_train, y_train, X_test, y_test, analysis
        )
        
        # Select best model
        best_model, best_performance = trainer.select_best_model(tf_results, sklearn_results)
        
        if best_model is None:
            logger.error("No valid models produced")
            return None, None
        
        # Save models with COMPLETE feature name preservation
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        model_save_dir = os.path.join(project_root, 'data', 'models')
        os.makedirs(model_save_dir, exist_ok=True)
        
        models_saved = []
        
        # Save TensorFlow model with COMPLETE metadata
        if tf_results is not None:
            tf_model_path = os.path.join(model_save_dir, f'{model_save_base_name}_{language_type.lower()}_tensorflow.h5')
            trainer.tensorflow_model.save(tf_model_path)
            
            tf_meta_path = os.path.join(model_save_dir, f'{model_save_base_name}_{language_type.lower()}_tensorflow_meta.pkl')
            tf_metadata = {
                'scaler': trainer.scaler,
                'label_encoder': trainer.label_encoder,
                'feature_names': trainer.feature_names,  # CRITICAL: Feature names saved here
                'accuracy': tf_results['accuracy'],
                'f1_macro': tf_results['f1_score'],
                'loss': tf_results['loss'],
                'language_type': language_type.upper(),
                'model_type': 'FIXED_TENSORFLOW',
                'dataset_analysis': analysis,
                'input_shape': tf_results['model'].input_shape,
                'output_classes': len(trainer.label_encoder.classes_),
                'class_names': list(trainer.label_encoder.classes_),
                'feature_count': len(trainer.feature_names),
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open(tf_meta_path, 'wb') as f:
                pickle.dump(tf_metadata, f)
            
            models_saved.append(f"TensorFlow: {tf_model_path}")
            models_saved.append(f"TensorFlow Meta: {tf_meta_path}")
        
        # Save Random Forest model with COMPLETE metadata
        if sklearn_results is not None:
            sklearn_model_path = os.path.join(model_save_dir, f'{model_save_base_name}_{language_type.lower()}_sklearn.pkl')
            sklearn_metadata = {
                'model': trainer.sklearn_model,
                'scaler': trainer.scaler,
                'feature_names': trainer.feature_names,  # CRITICAL: Feature names saved here
                'accuracy': sklearn_results['accuracy'],
                'f1_macro': sklearn_results['f1_score'],
                'language_type': language_type.upper(),
                'model_type': 'FIXED_RANDOM_FOREST',
                'dataset_analysis': analysis,
                'feature_importance': sklearn_results['feature_importance'].to_dict(),
                'n_features': len(trainer.feature_names),
                'class_names': list(trainer.sklearn_model.classes_),
                'feature_count': len(trainer.feature_names),
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open(sklearn_model_path, 'wb') as f:
                pickle.dump(sklearn_metadata, f)
            
            models_saved.append(f"Random Forest: {sklearn_model_path}")
        
        # Save standalone feature names file for debugging and compatibility
        feature_names_path = os.path.join(model_save_dir, f'{model_save_base_name}_{language_type.lower()}_feature_names.pkl')
        feature_names_metadata = {
            'feature_names': trainer.feature_names,
            'language_type': language_type.upper(),
            'total_features': len(trainer.feature_names),
            'first_10_features': trainer.feature_names[:10],
            'last_10_features': trainer.feature_names[-10:],
            'creation_timestamp': datetime.now().isoformat(),
            'training_session': f"{model_save_base_name}_{language_type.lower()}",
            'compatibility_note': 'These feature names MUST match exactly during inference for accurate predictions'
        }
        
        with open(feature_names_path, 'wb') as f:
            pickle.dump(feature_names_metadata, f)
        
        models_saved.append(f"Feature Names: {feature_names_path}")
        
        # Save best model as combined with COMPLETE metadata
        combined_path = os.path.join(model_save_dir, f'{model_save_base_name}_{language_type.lower()}.pkl')
        if best_model == 'tensorflow' and tf_results:
            best_metadata = tf_metadata.copy()
        else:
            best_metadata = sklearn_metadata.copy()
        
        best_metadata['best_model'] = best_model
        best_metadata['best_performance'] = best_performance
        best_metadata['all_available_models'] = []
        if tf_results: best_metadata['all_available_models'].append('tensorflow')
        if sklearn_results: best_metadata['all_available_models'].append('sklearn')
        
        with open(combined_path, 'wb') as f:
            pickle.dump(best_metadata, f)
        
        models_saved.append(f"Best Combined ({best_model}): {combined_path}")
        
        # Final logging with feature name confirmation
        logger.info(f"=== Training completed for {language_type} ===")
        for model_info in models_saved:
            logger.info(f"  ✅ Saved: {model_info}")
        
        logger.info(f"Performance Summary:")
        logger.info(f"  Best model: {best_model.upper()}")
        logger.info(f"  Best F1-score: {best_performance:.4f}")
        logger.info(f"  Dataset size: {analysis['total_samples']} samples")
        logger.info(f"  Features used: {len(trainer.feature_names)}")
        logger.info(f"  ✅ Feature names saved: YES (CRITICAL FOR INFERENCE)")
        
        # Log first and last feature names for verification
        logger.info(f"  First 5 features: {trainer.feature_names[:5]}")
        logger.info(f"  Last 5 features: {trainer.feature_names[-5:]}")
        
        # Verification message
        logger.info(f"🔑 CRITICAL: Feature names are now saved in model metadata")
        logger.info(f"🔑 Backend will use these names to align inference features")
        logger.info(f"🔑 This should fix the prediction accuracy issues")
        
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Error during training for {language_type}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
