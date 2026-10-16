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
FROM gcc:14.2.0

ENV DEBIAN_FRONTEND=noninteractive

COPY debian-backports.sources /etc/apt/sources.list.d/
COPY backports.pref /etc/apt/preferences.d/

WORKDIR /azp

RUN \
  echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes \
  && echo "KNP_IMAGE_VERSION=${KNP_IMAGE_VERSION}" >> /etc/environment \
  && export DEBIAN_VERSION=debian$(cat /etc/debian_version | sed 's/\..*//') \
  && apt-get update \
  && apt-get -y --no-install-recommends --show-progress install \
      apt-transport-https \
      build-essential \
      ca-certificates \
      cmake="3.31*" cmake-data="3.31*" \
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
      $(apt list|awk '/boost.*1.81.*-dev/ {print($1);}') \
  && apt-get clean

RUN rm -rf /tmp/*

COPY LICENSE.txt .
COPY NOTICE.txt .

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
