#!/usr/bin/env bash

set -e

CUDA_VERSION="12.9"

echo "[INFO] Checking for CUDA..."

if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] CUDA already installed:"
    nvcc --version
    exit 0
fi

echo "[WARN] CUDA not found. Installing..."

function install_cuda_debian() {
    export DEBIAN_VERSION=debian$(cat /etc/debian_version | sed 's/\..*//')
    echo "[INFO] Installing CUDA for ${DEBIAN_VERSION}..."

    CUDA_PKG_VERSION=$(echo ${CUDA_VERSION}|tr '.' '-')

    # cuda-keyring package adds correct repository.
    curl -fS https://developer.download.nvidia.com/compute/cuda/repos/${DEBIAN_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -f cuda-keyring_1.1-1_all.deb \
    && curl -sSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xA4B469963BF863CC" | \
       gpg --dearmor | tee /etc/apt/keyrings/cuda.gpg > /dev/null \
    && apt-get update \
    && apt-get install -y \
       cuda-minimal-build-${CUDA_PKG_VERSION} cuda-nvcc-${CUDA_PKG_VERSION} cuda-cudart-dev-${CUDA_PKG_VERSION}

    echo "[INFO] CUDA installation complete."
}

export PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

if [ "x$(uname -m)" = "xx86_64" ]; then
    install_cuda_debian

    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    else
        echo "[ERROR] CUDA installation failed"
        exit 1
    fi
else
    echo "Unsupported platform for CUDA: ${TARGETPLATFORM}"
fi

exec "$@"
