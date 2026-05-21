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

RUN \
  if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
    wget -qO- https://files.pvs-studio.com/etc/pubkey.txt | \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/viva64.gpg \
    && wget -O /etc/apt/sources.list.d/viva64.list \
               https://files.pvs-studio.com/etc/viva64.list \
    || exit 1; \
  fi

RUN \
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
      rapidjson-dev \
  && \
  if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
    echo "Installing PVS Studio..."; \
    apt-get -y --no-install-recommends install pvs-studio || exit 1; \
  fi \
  && apt-get clean \
  && cd /azp \
  && echo "Install Doxygen..." \
  && git clone --depth 1 --branch Release_1_13_2 --single-branch https://github.com/doxygen/doxygen.git \
  && cd doxygen \
  && cmake -G "Unix Makefiles" -DCMAKE_POSITION_INDEPENDENT_CODE=ON . \
  && cmake --build . --parallel 4 \
  && cmake --install . \
  && cd .. \
  && rm -rf doxygen

# COPY "${OCLINT_FILE}" .
RUN \
  pip install --break-system-packages gcovr pytest lizard \
  && pip uninstall --break-system-packages -y certifi \
  && rm -rf /tmp/*
#  && tar xf "${OCLINT_FILE}" \
#  && rsync -a oclint-22.02/* /usr/local/ \
COPY NOTICE.txt .
