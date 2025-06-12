# SilenBackEnd-ML

Proyek ini adalah backend untuk aplikasi berbasis Machine Learning (ML) yang menggunakan Flask sebagai server API dan berinteraksi dengan model ML yang dilatih. Proyek ini memuat logika untuk pemrosesan data, pelatihan model, evaluasi model, dan utilitas terkait deteksi tangan menggunakan MediaPipe. Data model yang digunakan diambil dari **Google Drive**.

## 📁 **Struktur Folder & Penjelasan:**
silenBackEnd-ML-main/ ← ROOT PROJECT
│
├── .dockerignore ← File untuk pengabaian Docker
├── .gitattributes ← Pengaturan atribut Git
├── .gitignore ← File untuk pengabaian Git
├── .railwayignore ← File untuk pengabaian Railway
├── Dockerfile ← File untuk konfigurasi Docker
├── README.md ← Dokumentasi proyek
├── app.py ← Flask API server
├── railway.json ← Konfigurasi untuk Railway
├── requirements.txt ← Python dependencies
│
├── data/ ← Folder data
│ └── models/ ← Tempat model ML disimpan
│ └── .gitkeep ← Menandai folder agar tetap tertrack
│
├── src/ ← Folder source code
│ ├── init.py ← Inisialisasi modul
│ ├── data_preprocessing/ ← Logika preprocessing data
│ │ ├── init.py
│ │ ├── augmentation.py ← Augmentasi gambar
│ │ ├── data_loader.py ← Pemrosesan dataset
│ │ └── feature_extractor.py ← Ekstraksi fitur tangan
│ ├── models/ ← Logika pelatihan dan evaluasi model
│ │ ├── init.py
│ │ ├── evaluate_model.py ← Evaluasi model
│ │ └── train_model.py ← Pelatihan model
│ └── utils/ ← Utilitas lainnya
│ ├── init.py
│ └── mediapipe_utils.py ← Menggunakan MediaPipe untuk deteksi tangan


## 🎯 **Penjelasan per Kategori:**

### **🤖 Machine Learning (Python)**

- **Bahasa:** Python 3.11
- **Framework & Library:**
  - TensorFlow 2.15.0
  - Keras 2.15.0
  - scikit-learn 1.3.0
  - MediaPipe 0.10.10
  - OpenCV 4.8.0.76 (headless)
  - NumPy 1.24.3
  - Pandas 2.0.3
  - Joblib 1.3.2
  - H5py 3.10.0
  - Pickle
  - ONNX 1.14.1
  - ONNX Runtime 1.16.0
  - TF2ONNX 1.15.1
  - Matplotlib 3.7.2
  - Seaborn 0.12.2
  - tqdm 4.66.1
- **Fungsi:**
  - **Data Preprocessing:** Menyiapkan dataset untuk pelatihan model, termasuk augmentasi gambar dan ekstraksi fitur tangan.
  - **Model Training:** Melatih model ML menggunakan dataset yang telah diproses.
  - **Model Evaluation:** Mengevaluasi kinerja model setelah pelatihan.
  - **Model Inference:** Menggunakan model yang telah dilatih untuk melakukan prediksi dari data input.

#### Struktur Folder `src/data_preprocessing/`:

- **augmentation.py:** Mengimplementasikan teknik augmentasi gambar untuk meningkatkan jumlah data latih.
- **data_loader.py:** Mengatur pemuatan dataset dari direktori dan mengonversi gambar menjadi format yang dapat digunakan untuk pelatihan.
- **feature_extractor.py:** Melakukan ekstraksi fitur dari gambar tangan menggunakan MediaPipe, untuk keperluan klasifikasi.

#### Struktur Folder `src/models/`:

- **train_model.py:** Berisi logika untuk melatih model ML menggunakan dataset yang telah diproses.
- **evaluate_model.py:** Menghitung dan menampilkan metrik evaluasi model, seperti akurasi, precision, recall, dll.

#### Struktur Folder `src/utils/`:

- **mediapipe_utils.py:** Utilitas untuk menggunakan MediaPipe untuk mendeteksi landmark tangan dalam gambar.

### **🌐 Frontend (JavaScript/React)**

- **Bahasa:** JavaScript (ES6+), HTML5, CSS3
- **Framework & Library:**
  - React.js 18.2.0
  - React DOM 18.2.0
  - React Router DOM 6.20.1
  - Vite 6.3.5
  - @vitejs/plugin-react 4.1.1
  - Tailwind CSS 3.3.5
  - PostCSS 8.4.31
  - Autoprefixer 10.4.16
  - @mediapipe/hands 0.4.1675469240
  - @tensorflow/tfjs 4.22.0
  - Axios 1.6.2
  - Lucide React 0.294.0
  - FontAwesome
  - ESLint 8.53.0
- **Fungsi:**
  - **User Interface:** Menyediakan antarmuka pengguna untuk interaksi dengan kamera dan pengunggahan gambar.
  - **Real-Time Translation:** Menampilkan hasil prediksi secara langsung dari model yang dilatih.
  - **API Communication:** Menghubungkan frontend dengan backend untuk mendapatkan hasil prediksi.

### **🖥️ Backend (Python Flask)**

- **Bahasa:** Python 3.11
- **Framework & Library:**
  - Flask 2.3.3
  - Flask-CORS 4.0.0
  - Gunicorn 21.2.0
  - Requests 2.31.0
  - Pillow 10.0.0
  - NumPy 1.24.3
  - Pandas 2.0.3
  - Base64
  - JSON
  - Pathlib
- **Deployment & Infrastructure:**
  - Docker
  - Railway
  - Python 3.11-slim-buster (base image)
  - libgl1-mesa-glx
  - libsm6, libxext6
  - libglib2.0-0
  - libfontconfig1, libxrender1
- **Fungsi:**
  - **Flask API Server:** Menyediakan endpoint API untuk menerima gambar dari frontend dan mengirimkan hasil prediksi.
  - **Model Loading & Inference:** Memuat model ML yang terlatih dari Google Drive dan melakukan prediksi berdasarkan gambar yang dikirimkan.

#### Struktur File `app.py`:

- **Flask App Initialization:** Menyiapkan server Flask dan konfigurasi.
- **API Endpoints:**
  - **/api/translate:** Menerima gambar dan mengembalikan hasil prediksi.
  - **/api/health:** Endpoint untuk memeriksa status API (misalnya, untuk memeriksa apakah API berjalan dengan baik).
  - **/api/load_model:** Mengunduh model ML yang disimpan di Google Drive dan menyiapkannya untuk prediksi.

## 🔄 **Alur Data:**
Frontend (React)
↓ 📸 Send image
Backend (Flask API)
↓ 🤖 Use ML model
ML Models (Python)
↓ ✨ Return prediction
Backend → Frontend
↓ 📱 Display result
User Interface


## 🚀 **Cara Menjalankan:**

### 1. Persiapkan lingkungan Python

```bash
# Install dependencies
pip install -r requirements.txt

# Melatih Model ML
Untuk melatih model, jalankan script main.py:
  python main.py              # Melatih model menggunakan dataset

# Jalankan Backend (Flask API)
Setelah model terlatih, jalankan server Flask untuk API:
python app.py               # Menjalankan Flask API (port 5000)
API akan berjalan di http://localhost:5000.

# Mengunduh Model dari Google Drive
Model yang digunakan oleh aplikasi akan diambil dari Google Drive saat menjalankan backend. Untuk mengunduh model ML, jalankan endpoint berikut:
# Mengambil model dari Google Drive dan mempersiapkan untuk inferensi
curl -X GET http://localhost:5000/api/load_model

# Jalankan Inferensi (Prediksi)
Setelah backend berjalan dan model berhasil dimuat, kamu dapat mengirimkan gambar untuk inferensi melalui endpoint API menggunakan POST request ke /api/translate.

Contoh request menggunakan curl:
curl -X POST http://localhost:5000/api/translate -F "image=@/path/to/image.jpg"
API akan mengembalikan hasil prediksi berdasarkan gambar yang dikirimkan.

# 🔗 Google Drive Data
Data yang digunakan oleh aplikasi tidak disimpan secara lokal, melainkan diakses melalui Google Drive. Untuk menggunakan aplikasi ini, pastikan untuk mengonfigurasi akses ke folder Google Drive yang sesuai untuk mendapatkan dataset dan model pelatihan.

Link folder Google Drive: Google Drive Folder
