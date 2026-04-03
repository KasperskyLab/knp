# Build image Dockerfile.
#
# © 2024-2026 AO Kaspersky Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG KNP_IMAGE_VERSION=2.0.0
ARG TARGETPLATFORM
FROM kasperskydh/knp-base-image:${KNP_IMAGE_VERSION}

ENV OCLINT_FILE=oclint-22.02-llvm-13.0.1-x86_64-linux-ubuntu-20.04.tar.gz \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /azp

RUN <<EOT
  echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes
  export DEBIAN_VERSION=debian$(cat /etc/debian_version | sed 's/\..*//')

#  && wget https://developer.download.nvidia.com/compute/cuda/repos/${DEBIAN_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb \
#  && dpkg -i cuda-keyring_1.1-1_all.deb \
#  && rm cuda-keyring_1.1-1_all.deb \
#  && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC \
#  && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${DEBIAN_VERSION}/x86_64/ /" | \
#    tee /etc/apt/sources.list.d/cuda-${DEBIAN_VERSION}-x86_64.list

  echo "BUILD IMAGE VERSION: ${KNP_IMAGE_VERSION}"
  echo "TARGETPLATFORM: ${TARGETPLATFORM}"

  if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then
    echo "Adding PVS Studio repo..."
    {
      wget -qO- https://files.pvs-studio.com/etc/pubkey.txt | gpg --dearmor -o /etc/apt/trusted.gpg.d/viva64.gpg \
      && wget -O /etc/apt/sources.list.d/viva64.list https://files.pvs-studio.com/etc/viva64.list
    } || exit 1
  fi

  {
    apt-get update \
    && apt-get -y --no-install-recommends --show-progress install \
      bandit \
      bison \
      cccc \
      clang \
      clang-tidy \
      cloc \
      cpplint \
      cppcheck \
      flake8 \
      flex \
      fonts-liberation \
      graphviz \
      gsfonts \
      mypy \
      libopencv-dev \
      pre-commit \
      protobuf-compiler \
      pylint \
      python3-dev \
      python3-build \
      python3-mypy \
      python3-setuptools \
      python3-types-requests \
      python3-wheel \
      rapidjson-dev
  } || { echo "Apt install failed!" >&2; exit 1; }

  if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then
    echo "Installing PVS Studio..."
    apt-get -y --no-install-recommends install pvs-studio || exit 1
  fi

  apt-get clean

  cd /azp
  echo "Install Doxygen..."

  # New Doxygen version.
  git clone --depth 1 --branch Release_1_13_2 --single-branch https://github.com/doxygen/doxygen.git \
  && cd doxygen \
  && cmake -G "Unix Makefiles" -DCMAKE_POSITION_INDEPENDENT_CODE=ON . \
  && cmake --build . --parallel 4 \
  && cmake --install . \
  && cd .. \
  && rm -rf doxygen
EOT

# COPY "${OCLINT_FILE}" .
RUN \
  pip3 install --break-system-packages gcovr pytest lizard \
  && pip3 uninstall --break-system-packages -y certifi
#  && tar xf "${OCLINT_FILE}" \
#  && rsync -a oclint-22.02/* /usr/local/ \
  && rm -rf /tmp/*
