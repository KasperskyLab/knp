#!/usr/bin/env bash

# @file start.sh
# @brief Entrypoint of the Docker containers.
# @kaspersky_support Artiom N.
# @date 22.04.2026
# @license Apache 2.0
# @copyright © 2026 AO Kaspersky Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

CUDA_VERSION="12.9"


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


if [ "x$(uname -m)" = "xx86_64" ]; then
    echo "[INFO] Checking for CUDA..."

    export PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    if command -v nvcc >/dev/null 2>&1; then
        echo "[OK] CUDA already installed:"
        nvcc --version
    else
        echo "[WARN] CUDA not found. Installing..."
        install_cuda_debian
    fi

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
