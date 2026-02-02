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
# Need to build tests.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

project(knp-tester VERSION "${KNP_VERSION}" LANGUAGES CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform tester and tests")

if (NOT KNP_BUILD_TESTS)
    message(STATUS "Building of tests disabled.")
    return()
endif()

set(GLOB_LIST "*.h" "*_test.cpp")

if (KNP_BUILD_CUDA)
    message(STATUS "Building CUDA tests enabled.")
    enable_language(CUDA)
    list(APPEND GLOB_LIST "*_test.cu")

    find_package(CUDAToolkit REQUIRED)
endif()

include(CTest)

include(memcheck)

enable_testing()

include(GoogleTest)

file(GLOB_RECURSE COMMON_SOURCE CONFIGURE_DEPENDS ${GLOB_LIST})

add_executable("${PROJECT_NAME}" ${COMMON_SOURCE} tester.cpp utility.cpp)

target_include_directories("${PROJECT_NAME}"
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/common"
    PRIVATE "${GTEST_DIR}/googletest/include"
    PRIVATE "${GTEST_DIR}/googlemock/include"
)

#knp_get_hdf5_target(HDF5_LIB)

target_link_libraries("${PROJECT_NAME}" PRIVATE KNP::BaseFramework::CoreStatic KNP::Backends::CPUSingleThreaded KNP::Backends::CPUMultiThreaded
                                                KNP::Backends::CPU::ThreadPool)

if (KNP_BUILD_CUDA)
    target_link_libraries("${PROJECT_NAME}" PRIVATE knp-gpu-cuda-backend_static CUDA::cudart)  # KNP::Backends::CUDA
endif()

target_link_libraries("${PROJECT_NAME}" PRIVATE gtest gtest_main spdlog::spdlog_header_only) #  HighFive

add_dependencies("${PROJECT_NAME}" knp-base-framework-core_static)

if (${BUILD_GMOCK})
    target_link_libraries("${PROJECT_NAME}" PRIVATE gmock gmock_main)
endif()

gtest_discover_tests("${PROJECT_NAME}"
    # Set a working directory to find test data via paths relative to the project root.
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if (KNP_ENABLE_COVERAGE)
    message(STATUS "Coverage enabled.")

    set(COV_EXECUTABLE "${PROJECT_NAME}")
    #set(COV_EXECUTABLE "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    set(COV_NAME "${PROJECT_NAME}_coverage")
    set(COV_DEPENDENCIES "${PROJECT_NAME}")
    set(COV_BASE_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set(COV_EXCLUDES "/usr/*;third-party/*")

    set(CODE_COVERAGE_VERBOSE ON)

    if (GCOVR_PATH)
        set(GCOVR_ADDITIONAL_ARGS --gcov-ignore-parse-errors)
        setup_target_for_coverage_gcovr_xml(
            NAME "${COV_NAME}"
            EXECUTABLE ${COV_EXECUTABLE}
            DEPENDENCIES ${COV_DEPENDENCIES}
            BASE_DIRECTORY "${COV_BASE_DIRECTORY}"
        )
        message(VERBOSE "Gcovr selected.")
    elseif (LCOV_PATH)
        setup_target_for_coverage_lcov(
            NAME "${COV_NAME}"
            EXECUTABLE ${COV_EXECUTABLE}
            DEPENDENCIES ${COV_DEPENDENCIES}
            BASE_DIRECTORY "${COV_BASE_DIRECTORY}"
            EXCLUDE ${COV_EXCLUDES}
            LCOV_ARGS
                --ignore-errors empty,inconsistent
        )
        message(VERBOSE "Lcov selected.")
    elseif (FASTCOV_PATH)
        setup_target_for_coverage_fastcov(
            NAME "${COV_NAME}"
            EXECUTABLE ${COV_EXECUTABLE}
            DEPENDENCIES ${COV_DEPENDENCIES}
            BASE_DIRECTORY "${COV_BASE_DIRECTORY}"
        )
        message(VERBOSE "Fastcov selected.")
    else()
        message(WARNING "Coverage cannot be enabled.")
    endif()
endif()


set(CTEST_MEMORYCHECK_COMMAND "/usr/bin/valgrind")
set(CTEST_MEMORYCHECK_COMMAND_OPTIONS "--tool=memcheck --leak-check=full --undef-value-errors=yes --error-limit=no --time-stamp=yes --trace-children=yes --track-fds=yes --track-origins=yes --gen-suppressions=all --undef-value-errors=yes --error-exitcode=1")

add_custom_target(test_with_memcheck
    COMMAND ${CMAKE_CTEST_COMMAND}
        --force-new-ctest-process --test-action memcheck
        --rerun-failed --output-on-failure
    COMMAND cat "Testing/Temporary/MemoryChecker.*.log"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/knp/tests")

if (KNP_BUILD_CUDA)
    set_property(TARGET "${PROJECT_NAME}" PROPERTY CUDA_ARCHITECTURES 60)
    target_link_options("${PROJECT_NAME}" PRIVATE "-Wl,--allow-shlib-undefined")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options("${PROJECT_NAME}" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
