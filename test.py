import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

from src.data_preprocessing.feature_extractor import create_standardized_features
from src.data_preprocessing.feature_extractor import get_expected_feature_names

# Load model dan feature names
model = joblib.load("data/models/sign_language_model_sibi_sklearn.pkl")
feature_meta = joblib.load("data/models/sign_language_model_sibi_feature_names.pkl")
expected_feature_names = feature_meta["feature_names"]

# Load gambar
image = cv2.imread("f (19).png")
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("❌ Tidak ada tangan terdeteksi.")
    else:
        print("✅ Tangan terdeteksi.")

        # Gabungkan semua landmark jadi 126 angka (2 tangan × 21 landmark × 3 koordinat)
        all_landmarks = [0.0] * 126  # Init dengan nol
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            for j, lm in enumerate(hand_landmarks.landmark):
                index = i * 63 + j * 3
                all_landmarks[index] = lm.x
                all_landmarks[index + 1] = lm.y
                all_landmarks[index + 2] = lm.z

        # Ekstrak fitur standar
        features_df = create_standardized_features(all_landmarks)

        # Reorder kolom fitur sesuai dengan model
        X = features_df[expected_feature_names]

        # Prediksi
        pred = model.predict(X)[0]
        print(f"📘 Prediksi huruf: {pred}")
