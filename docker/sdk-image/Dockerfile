# KNP SDK image Dockerfile.
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
ARG KNP_VERSION=2.0.0
ARG TARGETPLATFORM

FROM kasperskydh/knp-base-image:${KNP_IMAGE_VERSION}

WORKDIR /knp

RUN \
  apt-get update \
  && apt-get -y --no-install-recommends --show-progress install \
    python3-dev \
    python3-build \
    python3-mypy \
    python3-setuptools \
    python3-types-requests \
    python3-wheel \
    zsh \
  && apt-get -y --no-install-recommends upgrade \
  && apt-get clean \
  && chsh -s $(which zsh) \
  && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
  && cd /knp

COPY NOTICE.txt .
COPY *.deb /knp/

RUN DEBIAN_FRONTEND=noninteractive dpkg -i \
  /knp/knp-cpp-framework_2.0.0_amd64.deb /knp/knp-cpp-framework-dev_2.0.0_amd64.deb \
  /knp/knp-documentation_2.0.0_all.deb knp-examples_2.0.0_amd64.deb \
  /knp/knp-cpu-single-threaded-backend_2.0.0_amd64.deb /knp/knp-cpu-single-threaded-backend-dev_2.0.0_amd64.deb \
  /knp/knp-cpu-multi-threaded-backend_2.0.0_amd64.deb /knp/knp-cpu-multi-threaded-backend-dev_2.0.0_amd64.deb \
  /knp/knp-examples_2.0.0_amd64.deb \
  /knp/knp-python3-framework_2.0.0_amd64.deb
