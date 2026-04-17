#!/usr/bin/env bash

set -e

echo "[INFO] Checking for CUDA..."

if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] CUDA already installed:"
    nvcc --version
    exit 0
fi

echo "[WARN] CUDA not found. Installing..."

function install_cuda_debian() {
    echo "[INFO] Installing CUDA for Debian..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit

    echo "[INFO] CUDA installation complete."
}

install_cuda_debian

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "[ERROR] CUDA installation failed"
    exit 1
fi
