# Dockerfile untuk aplikasi Sign Language API (Railway/Production)
# Menggunakan base image python:3.11-slim-buster untuk kompatibilitas yang lebih baik
FROM python:3.11-slim-buster 

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV FLASK_ENV=production

# Install system dependencies yang diperlukan oleh OpenCV (cv2)
# libgl1-mesa-glx menyediakan libGL.so.1
# libsm6 dan libxext6 juga sering dibutuhkan oleh OpenCV di lingkungan headless
# libglib2.0-0 menyediakan libgthread-2.0.so.0
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libfontconfig1 \  
    libxrender1 && \  
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set work directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Menyalin folder src (termasuk feature_extractor.py jika ada)
COPY src/ src/

# Menyalin folder models yang berisi file model (.pkl, .h5, .joblib)
COPY data/models/ data/models/

# Menyalin file aplikasi utama (app.py)
COPY app.py .

# Tambahkan baris ini untuk debugging struktur file:
RUN ls -R /app 

# Create dummy directories (ini mungkin sudah tidak sepenuhnya diperlukan
# jika folder data/models dan src/ sudah disalin, tapi tidak ada salahnya)
RUN mkdir -p data/models src/data_preprocessing

# Expose port
EXPOSE $PORT

# Simple health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Start command - gunakan shell form
CMD python3 app.py
