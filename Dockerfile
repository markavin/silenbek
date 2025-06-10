# =========================================================
# DOCKERFILE KHUSUS UNTUK ML - OPTIMIZED SIZE
# =========================================================

FROM python:3.12-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas3 \
    liblapack3 \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages dengan optimasi khusus
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
    -r requirements.txt && \
    # Bersihkan cache pip
    pip cache purge

# Copy aplikasi
COPY . .

# Bersihkan file tidak perlu
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + && \
    find . -name "*.pyo" -delete && \
    find . -name "tests" -type d -exec rm -rf {} + && \
    find . -name "test_*" -delete

# Set environment variables untuk optimasi
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "app.py"]
