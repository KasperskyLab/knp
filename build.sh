#!/bin/bash

# @file build.sh
# @kaspersky_support Artiom N.
# @license Apache 2.0
# @copyright © 2024-2025 AO Kaspersky Lab
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


VERBOSE=0
RELEASE=0
SINGLE_PROCESS=0
CMAKE_BINARY=cmake
CMAKE_ADD_OPTS=${CMAKE_ADD_OPTS:-""}
MAKE_ADD_OPTS=${MAKE_ADD_OPTS:-""}

CURDIR="$(dirname "$(readlink -f "$0")")"

usage() { echo "Usage: $0 [-v] [-r] [-c] [-s]" 1>&2; }


build()
{
  if [[ $VERBOSE == 1 ]]; then
    MAKE_ADD_OPTS="${MAKE_ADD_OPTS} VERBOSE=1"
  fi

  if [[ $RELEASE == 1 ]]; then
    echo "Building release..."
    BUILD_DIR="build_release"
    CMAKE_ADD_OPTS="${CMAKE_ADD_OPTS} -DCMAKE_BUILD_TYPE=Release"
  else
    echo "Building debug..."
    CMAKE_ADD_OPTS="${CMAKE_ADD_OPTS} -DCMAKE_BUILD_TYPE=Debug"
    BUILD_DIR="build"
  fi

  if [[ ${CLEAN} == 1 ]]; then
    BUILD_ADD_OPTS="${BUILD_ADD_OPTS} --clean-first"
  fi

  if [[ ${SINGLE_PROCESS} == 0 ]]; then
    BUILD_ADD_OPTS="${BUILD_ADD_OPTS} --parallel $(nproc)"
  fi

  set -e
  set -o xtrace

  # mkdir build 2> /dev/null

  export CCACHE_DISABLE=1
  export ${MAKE_ADD_OPTS}
#        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  "${CMAKE_BINARY}" \
        -DKNP_BUILD_TESTS=true \
        -DCMAKE_INSTALL_PREFIX="install" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
        ${CMAKE_ADD_OPTS} \
        -B "${BUILD_DIR}" \
        -S . && \
  "${CMAKE_BINARY}" --build "${BUILD_DIR}" ${BUILD_ADD_OPTS}
}


while getopts "rcvsh" o; do
  case "${o}" in
    h)
      usage
      exit 0
    ;;
    c)
      CLEAN=1
    ;;
    v)
      VERBOSE=1
    ;;
    r)
      RELEASE=1
    ;;
    s)
      SINGLE_PROCESS=1
    ;;
    *)
      shift $((OPTIND-1))
      CMAKE_ADD_OPTS="${CMAKE_ADD_OPTS} $*"
      break;
    ;;
  esac
done

shift $((OPTIND-1))

build
