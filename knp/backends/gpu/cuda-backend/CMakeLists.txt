#[[
© 2025 AO Kaspersky Lab

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

if (NOT KNP_BUILD_CUDA)
    message(STATUS "Building CUDA backend disabled.")
    return()
endif()

project(knp-gpu-cuda-backend VERSION "${KNP_VERSION}" LANGUAGES CUDA C CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform NVIDIA CUDA GPU backend"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(${PROJECT_NAME}_PUBLIC_INCLUDE_DIR "knp/backends/gpu/cuda")

include(GNUInstallDirs)
include(clang-tidy)
include(knp-functions)

find_package(CUDAToolkit REQUIRED)
find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS system filesystem REQUIRED)

file(GLOB_RECURSE ${PROJECT_NAME}_headers include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/*.h)
file(GLOB_RECURSE ${PROJECT_NAME}_cuda_headers include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/*.cuh)

knp_add_library("${PROJECT_NAME}"
    BOTH
        impl/backend.cu
        impl/backend_impl.cu
        impl/uid.cu
        impl/gpu_cuda_device.cu
        impl/cuda_bus/messaging.cu
        impl/cuda_bus/message_bus.cu
        impl/cuda_bus/subscription.cu
        impl/cuda_lib/kernels.cu
        impl/cuda_lib/kernels.cu
        # impl/get_network.cpp
        ${${PROJECT_NAME}_headers}
        ${${PROJECT_NAME}_cuda_headers}
    ALIAS KNP::Backends::CUDA
    LINK_PRIVATE
        Boost::headers ${Boost_LIBRARIES} spdlog::spdlog_header_only CUDA::cudart CUDA::cuda_driver CUDA::nvml
    LINK_PUBLIC
        KNP::Core
)

set_target_properties(
        "${PROJECT_NAME}"
        PROPERTIES
        VERSION "${CMAKE_PROJECT_VERSION}"
        SOVERSION "${CMAKE_PROJECT_VERSION_MAJOR}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options("${PROJECT_NAME}" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -rdc=true>)
target_compile_options("${PROJECT_NAME}_static" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr -rdc=true>)
set_property(TARGET "${PROJECT_NAME}" PROPERTY CUDA_ARCHITECTURES 60)
set_property(TARGET "${PROJECT_NAME}_static" PROPERTY CUDA_ARCHITECTURES 60)

if (KNP_INSTALL)
    knp_get_component_name("${PROJECT_NAME}" COMPONENT_NAME)

#    install(TARGETS "${PROJECT_NAME}"
#            EXPORT "${PROJECT_NAME}"
#            COMPONENT "${COMPONENT_NAME}"
#            LIBRARY)

#    install(TARGETS "${PROJECT_NAME}_static"
#            EXPORT "${PROJECT_NAME}_static"
#            COMPONENT "${COMPONENT_NAME}-dev"
#            ARCHIVE)

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
