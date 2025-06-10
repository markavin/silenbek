# Multi-stage build untuk Railway (sangat optimal)
FROM python:3.11-slim as base

# Install system dependencies minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ===== STAGE 1: Build dependencies =====
FROM base as builder

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies ke virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies dengan optimasi maksimal
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# ===== STAGE 2: Runtime image =====
FROM base as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV FLASK_ENV=production
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment dari builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy aplikasi code
COPY app.py .
COPY requirements.txt .

# Buat direktori untuk models tapi kosong dulu
RUN mkdir -p data/models src/data_preprocessing

# Buat dummy files untuk menghindari import error
RUN echo 'def extract_features(df, perform_selection=False): return df' > src/data_preprocessing/feature_extractor.py && \
    touch src/data_preprocessing/__init__.py && \
    touch src/__init__.py

# Buat user non-root untuk security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE $PORT

# Health check ringan
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Command untuk produksi
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "--threads", "2", "--worker-class", "gthread", "app:app"]
