# silent-ml/src/models/evaluate_model.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_evaluate_model(model_path, X_test, y_test):
    """
    Memuat model yang sudah dilatih dan mengevaluasinya.
    
    Args:
        model_path (str): Path ke model yang disimpan.
        X_test (pd.DataFrame): Fitur data pengujian.
        y_test (pd.Series): Label data pengujian.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return

    y_pred = model.predict(X_test)

    print(f"\n--- Evaluation Report for {model_path} ---")
    print(classification_report(y_test, y_pred))
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.2f}")

    # Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix for {os.path.basename(model_path)}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()