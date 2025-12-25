#[[
© 2024-2025 AO Kaspersky Lab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

project(knp-cpu-backends-library VERSION "${KNP_VERSION}" LANGUAGES C CXX)

find_package(Boost ${KNP_BOOST_MIN_VERSION} REQUIRED)

include(clang-tidy)
include(knp-functions)

file(GLOB_RECURSE ${PROJECT_NAME}_headers include/knp/backends/cpu-library/*.h)

knp_add_library("${PROJECT_NAME}"
        INTERFACE
        ${${PROJECT_NAME}_headers})
add_library(KNP::Backends::CPU::Library ALIAS "${PROJECT_NAME}")

source_group(source REGULAR_EXPRESSION "impl/.*")
source_group(headers FILES ${${PROJECT_NAME}_headers} REGULAR_EXPRESSION "include/.*")
source_group(headers REGULAR_EXPRESSION "impl/.*")

target_include_directories("${PROJECT_NAME}" INTERFACE ${Boost_INCLUDE_DIRS})
target_include_directories("${PROJECT_NAME}" INTERFACE "impl")

target_link_libraries("${PROJECT_NAME}" INTERFACE Boost::headers spdlog::spdlog_header_only)
target_link_libraries("${PROJECT_NAME}" INTERFACE KNP::Core KNP::Devices::CPU)

# Internal library, used by backends.
# This doesn't require installation.
