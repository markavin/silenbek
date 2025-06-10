# Dockerfile ultra-sederhana untuk Railway
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV FLASK_ENV=production

# Install system dependencies minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy app
COPY app.py .

# --- PENTING: Tambahkan ini untuk menyalin folder model Anda ---
# Pastikan folder 'data/models' ada di root proyek backend lokal Anda
# dan berisi file model Anda (mis. your_model_file.h5)
COPY data/models/ data/models/
# Jika file model Anda langsung di root backend (tidak disarankan):
# COPY your_model_file.h5 data/models/
# Atau jika Anda memiliki struktur yang lebih kompleks, sesuaikan path ini.

# Buat dummy directories (opsional, karena model sudah disalin)
# RUN mkdir -p data/models src/data_preprocessing # Ini tidak perlu lagi membuat data/models jika sudah dicopy

# Expose port
EXPOSE $PORT

# Simple health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Start command - SIMPLE
CMD ["python", "app.py"]
