# Base image Dockerfile.
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
FROM gcc:14.1.0

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /azp

RUN <<EOT
  echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes
  export DEBIAN_VERSION=debian$(cat /etc/debian_version | sed 's/\..*//')

  echo "BASE IMAGE VERSION: ${KNP_IMAGE_VERSION}"
  echo "TARGETPLATFORM: ${TARGETPLATFORM}"

  {
    apt-get update \
    && apt-get -y --no-install-recommends --show-progress install \
      apt-transport-https \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      gdbserver \
      git \
      gnupg-agent \
      jq \
      libcurl4 \
      libunwind8 \
      netcat-openbsd \
      net-tools \
      libopencv-dev \
      p7zip-full \
      python3 \
      python3-opencv \
      python3-pip \
      python3-requests \
      python3-types-requests \
      software-properties-common \
      ssh \
      strace \
      tar \
      valgrind \
      wget \
      zip \
      $(apt list|awk '/boost.*1.81.*-dev/ {print($1);}')
  } || { echo "Apt install failed!" >&2; exit 1; }

  apt-get clean

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
  && add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        mantic \
        stable" \
  && export VERSION=$(curl --silent https://api.github.com/repos/docker/compose/releases/latest | jq .name -r) \
  && export DESTINATION=/usr/local/bin/docker-compose \
  && curl -L https://github.com/docker/compose/releases/download/${VERSION}/docker-compose-`uname -s`-`uname -m` -o "${DESTINATION}" \
  && chmod +x ${DESTINATION} \
  && ln -sf ${DESTINATION} /usr/bin/docker-compose
EOT

RUN rm -rf /tmp/*

COPY LICENSE.txt .
COPY NOTICE.txt .
