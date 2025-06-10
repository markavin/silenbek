# =========================================================
# DOCKERFILE OPTIMIZED UNTUK RAILWAY (TARGET: <4GB)
# Python 3.12, Debian-based (slim) dengan optimasi ukuran
# =========================================================

FROM python:3.12-slim

# Set environment variables untuk mengurangi ukuran
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependensi sistem minimal (hanya yang benar-benar diperlukan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools (akan dihapus nanti)
    build-essential \
    pkg-config \
    cmake \
    git \
    # Library untuk OpenCV dan ML (minimal)
    libjpeg62-turbo-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    # Python development
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first untuk layer caching
COPY requirements.txt .

# Install Python dependencies dengan optimasi
RUN pip install --no-cache-dir -r requirements.txt \
    # Hapus cache dan temporary files
    && pip cache purge \
    && find /usr/local/lib/python3.12 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.12 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove build dependencies untuk menghemat space
RUN apt-get purge -y --auto-remove \
    build-essential \
    pkg-config \
    cmake \
    git \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy aplikasi
COPY . .

# Remove unnecessary files
RUN find . -name "*.pyc" -delete \
    && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find . -name "*.pyo" -delete \
    && find . -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find . -name "*.md" -delete 2>/dev/null || true \
    && find . -name "*.txt" ! -name "requirements.txt" -delete 2>/dev/null || true

# Create non-root user untuk keamanan
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')" || exit 1

# Expose port
EXPOSE 5000

# Run aplikasi
CMD ["python", "app.py"]
