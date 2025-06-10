# =========================================================
# DOCKERFILE UNTUK APLIKASI PYTHON DENGAN DEPENDENSI ML
# Python 3.12, Debian-based (slim)
# =========================================================

# --- PILIH HANYA SATU BARIS 'FROM' BERIKUT INI DENGAN MENGHAPUS TANDA '#' ---

# OPSI 1 (Direkomendasikan Pertama): Base image Python 3.12 slim
# Ini menggunakan distribusi Debian terbaru (Bookworm) secara default untuk 3.12-slim
FROM python:3.12-slim

# OPSI 2 (Alternatif jika OPSI 1 gagal): Base image Python 3.12 slim di atas Bookworm
# Ini secara eksplisit menentukan distribusi Debian Bookworm
# # FROM python:3.12-slim-bookworm

# =========================================================

# Instal dependensi sistem yang umum dibutuhkan oleh pustaka ML
# seperti OpenCV, TensorFlow, dan untuk proses build (compilers, dll.).
# Perintah ini menggunakan apt-get karena base image adalah Debian/Ubuntu.
# ...
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbbmalloc2 \
    libtbb-dev \
    libv4l-dev \
    libgtk2.0-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    git \
    cmake \
    # Hapus cache apt setelah instalasi untuk mengurangi ukuran image
    && rm -rf /var/lib/apt/lists/*

# ---- Solusi Agresif untuk Masalah 'distutils' di Python 3.12 ----
# Pastikan pip, setuptools, dan wheel di lingkungan GLOBAL sudah mutakhir
# sebelum virtual environment dibuat atau paket lain diinstal.
RUN pip install --no-input --upgrade pip setuptools wheel

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan PATH untuk virtual environment
# Ini memastikan executable dari venv diprioritaskan di PATH
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Buat virtual environment untuk mengisolasi dependensi proyek
RUN python -m venv --copies /opt/venv

# AKTIFKAN VIRTUAL ENVIRONMENT dan LAKUKAN UPGRADE LAGI DI DALAMNYA
# Ini adalah lapisan pengaman tambahan untuk memastikan setuptools di dalam venv
# juga mutakhir sebelum menginstal requirements.
RUN . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

# Salin file requirements.txt ke dalam container.
# Ini dilakukan sebelum salin kode aplikasi untuk memanfaatkan Docker layer caching.
COPY requirements.txt .

# Sekarang, install semua dependensi Python yang tercantum di requirements.txt.
# Karena pip/setuptools sudah di-upgrade, masalah distutils seharusnya teratasi.
RUN . /opt/venv/bin/activate && \
    pip install -r requirements.txt

# Salin sisa kode aplikasi dari host ke dalam container
# Pastikan ini dilakukan setelah instalasi dependensi, sehingga perubahan kode
# tidak memicu instalasi ulang dependensi jika requirements.txt tidak berubah.
COPY . .

# Atur command yang akan dijalankan saat kontainer dimulai.
# Sesuaikan dengan entry point utama aplikasi Backend Flask Anda.
# Contoh: Jika file Flask Anda bernama app.py dan dijalankan langsung
CMD ["python", "app.py"]
