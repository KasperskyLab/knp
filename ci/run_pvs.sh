#!/bin/sh -x

# @file run_pvs.sh
# @kaspersky_support Artiom N.
# @license Apache 2.0
# @copyright © 2025 AO Kaspersky Lab
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


PVS_LICENSE="${1} ${2}"
BUILD_DIR="${3:-build}"

pvs-studio-analyzer credentials ${PVS_LICENSE}
cmake --build "${BUILD_DIR}" --parallel --target pvs-analyze
cat PVS-Studio.log
