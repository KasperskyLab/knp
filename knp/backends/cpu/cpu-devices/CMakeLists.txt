#[[
© 2024 AO Kaspersky Lab

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

# Required for the linters.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(knp-cpu-devices VERSION "${KNP_VERSION}" LANGUAGES C CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform device parameters library"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

include(GNUInstallDirs)
include(clang-tidy)
include(knp-functions)

set(${PROJECT_NAME}_PUBLIC_INCLUDE_DIR "knp/devices")

if ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86.*)|(X86.*)|(amd64)|(AMD64)")
    set(${PROJECT_NAME}_CPU_SOURCE impl/x86_cpu.cpp impl/x86_cpu_power.cpp impl/x86_cpu_power.h)
    set(${PROJECT_NAME}_ADD_LINK_LIBRARIES PCM_STATIC)
else()
    message(WARNING "This CPU is unknown and not totally supported.")
    set(${PROJECT_NAME}_CPU_SOURCE impl/unknown_cpu.cpp)
endif()

knp_add_library("${PROJECT_NAME}"
    STATIC

    ${${PROJECT_NAME}_CPU_SOURCE}
    include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/cpu.h

    LINK_PRIVATE
        spdlog::spdlog_header_only ${${PROJECT_NAME}_ADD_LINK_LIBRARIES}
        KNP::Core
    ALIAS KNP::Devices::CPU
)

set_target_properties(
        "${PROJECT_NAME}"
        PROPERTIES
        VERSION "${CMAKE_PROJECT_VERSION}"
        SOVERSION "${CMAKE_PROJECT_VERSION_MAJOR}")


if (KNP_INSTALL)
    set(COMPONENT_NAME "cpp-framework")

    install(TARGETS "${PROJECT_NAME}"
            EXPORT "${PROJECT_NAME}"
            COMPONENT "${COMPONENT_NAME}-dev"
            ARCHIVE)

    install(DIRECTORY "include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}"
            COMPONENT "${COMPONENT_NAME}-dev"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/knp"
            FILES_MATCHING PATTERN "*.h")
endif()
