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

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

project(knp-cpu-multi-threaded-backend VERSION "${KNP_VERSION}" LANGUAGES C CXX
        DESCRIPTION "CPU multi-threaded backend"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

set(${PROJECT_NAME}_PUBLIC_INCLUDE_DIR "knp/backends/cpu-multi-threaded")

include(GNUInstallDirs)
include(clang-tidy)
include(knp-functions)

find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS system REQUIRED)

file(GLOB_RECURSE ${PROJECT_NAME}_headers include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/*.h)
source_group(source REGULAR_EXPRESSION "impl/.*")
source_group(headers REGULAR_EXPRESSION "include/.*")

knp_add_library("${PROJECT_NAME}"
    BOTH
        impl/backend.cpp
        impl/get_network.cpp
        ${${PROJECT_NAME}_headers}
    ALIAS KNP::Backends::CPUMultiThreaded
    LINK_PRIVATE
        Boost::headers spdlog::spdlog_header_only
        KNP::Backends::CPU::Library KNP::Backends::CPU::ThreadPool
    LINK_PUBLIC
        KNP::Devices::CPU
)

set_target_properties(
        "${PROJECT_NAME}"
        PROPERTIES
        VERSION "${CMAKE_PROJECT_VERSION}"
        SOVERSION "${CMAKE_PROJECT_VERSION_MAJOR}")

if (KNP_INSTALL)
    knp_get_component_name("${PROJECT_NAME}" COMPONENT_NAME)

    install(TARGETS "${PROJECT_NAME}"
            EXPORT "${PROJECT_NAME}"
            COMPONENT "${COMPONENT_NAME}"
            LIBRARY)

    install(TARGETS "${PROJECT_NAME}_static"
            EXPORT "${PROJECT_NAME}_static"
            COMPONENT "${COMPONENT_NAME}-dev"
            ARCHIVE)

    install(DIRECTORY "include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}"
            COMPONENT "${COMPONENT_NAME}-dev"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/knp/backends"
            FILES_MATCHING PATTERN "*.h")

    knp_packaging_set_parameters("${COMPONENT_NAME}" "${PROJECT_NAME}"
            DESCRIPTION "${PROJECT_DESCRIPTION}"
            DEPENDS "libboost-all (= ${Boost_VERSION})" "libzmq"
            PROVIDES "knp-backend")

    knp_packaging_set_parameters("${COMPONENT_NAME}-dev" "${PROJECT_NAME}"
            DESCRIPTION "${PROJECT_DESCRIPTION} development package: C++ headers and CMake modules"
            DEPENDS "${COMPONENT_NAME}")
endif()
