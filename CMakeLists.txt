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
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHED)

include_guard(GLOBAL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

message(STATUS "cmake version ${CMAKE_VERSION}")
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    # cmake > 4.0 does not support projects with minimal versions < 3.5
    message(STATUS "Changing cmake minimum version to 3.5 globally")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE INTERNAL "")
endif()

include(CMakeDependentOption)
include(third-party)
include(version)

knp_get_version()

if(NOT KNP_VERSION)
    message(FATAL_ERROR "Kaspersky Neuromorphic Platform version is mandatory for build, but it cannot be determined.")
endif()

enable_language(CXX)

if (POLICY CMP0091)
    # Enable MSVS library selection using property.
    cmake_policy(SET CMP0091 NEW)
endif()

if (POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

project(knp VERSION "${KNP_VERSION}"
        DESCRIPTION "Kaspersky Neuromorphic Platform"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

set(KNP_BOOST_MIN_VERSION "1.74.0")
set(KNP_ROOT_DIR "${PROJECT_SOURCE_DIR}" CACHE STRING "Kaspersky Neuromorphic Platform root directory")

if (MSVC)
    add_compile_options(/EHsc /MP /MDd)
    # add_link_options(/LTCG:INCREMENTAL)  # /NODEFAULTLIB:LIBCMTD
    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING=1 _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS=1)
endif()

foreach (boost_ver RANGE 81 89)
    list(APPEND Boost_ADDITIONAL_VERSIONS "1.${boost_ver}")
endforeach()

if (NOT Boost_ROOT)
    if (DEFINED ENV{Boost_ROOT})
        set(Boost_ROOT "$ENV{Boost_ROOT}")
    elseif(DEFINED ENV{BOOST_ROOT})
        set(Boost_ROOT "$ENV{BOOST_ROOT}")
    endif()
endif()

if (Boost_ROOT)  # WIN32?
    # WARNING: Do not change variable case.
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_ALL_DYN_LINK OFF)

    if (NOT WIN32)
        set(Boost_USE_STATIC_RUNTIME ON)
    endif()

    # Boost detection setup.
    if(NOT BOOST_INCLUDE_DIR)
        set(BOOST_INCLUDE_DIR "${BOOST_ROOT}")
    endif()

    if (NOT BOOST_LIBRARYDIR)
        if(EXISTS "${Boost_ROOT}/stage/lib")
            set(BOOST_LIBRARYDIR "${Boost_ROOT}/stage/lib")
        elseif(EXISTS "${Boost_ROOT}/stage")
            set(BOOST_LIBRARYDIR "${Boost_ROOT}/stage")
        else()
            file(GLOB _BOOST_LIB_DIRS LIST_DIRECTORIES true "${Boost_ROOT}/lib*-*-*")
            if (_BOOST_LIB_DIRS)
                list(GET _BOOST_LIB_DIRS 0 BOOST_LIBRARYDIR)
            endif()
            # set(Boost_LIB_PREFIX "")
        endif()
    endif()

    message(STATUS "Boost library directory: ${BOOST_LIBRARYDIR}.")
endif()

# set(Boost_DEBUG ON)
find_package(Boost ${KNP_BOOST_MIN_VERSION} REQUIRED)

if (MSVC)
    message(STATUS "Adding \"${BOOST_LIBRARYDIR}\" to LIBPATH")
    add_link_options("/LIBPATH:${BOOST_LIBRARYDIR}")
endif()


#
# Kaspersky Neuromorphic Platform build option set.
#

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(KNP_BUILD_AUTONOMOUS TRUE)
else()
    # Build is not from source directory, it is possible to specify it via add_subdirectory().
    set(KNP_BUILD_AUTONOMOUS FALSE)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(KNP_ENABLE_AVX_DEFAULT ON)
    endif()
else()
    set(KNP_ENABLE_AVX_DEFAULT OFF)
endif()

if (KNP_BUILD_AUTONOMOUS OR KNP_MAINTAINER_BUILD)
    find_package(Doxygen OPTIONAL_COMPONENTS dot)
    if(NOT DOXYGEN_FOUND)
        message(WARNING "Doxygen was not found. Documentation build will be disabled by default.")
    endif()
endif()

find_program(_STUBGEN stubgen NO_CACHE)

if ("${_STUBGEN}" STREQUAL "_STUBGEN-NOTFOUND")
    message(WARNING "MyPy stubgen was not found. Please install MyPy (see requirements.txt). Python framework build will be disabled.")
    set(KNP_PYTHON_FRAMEWORK_BUILD_DEFAULT OFF)
else()
    set(KNP_PYTHON_FRAMEWORK_BUILD_DEFAULT ON)
endif()

cmake_dependent_option(KNP_BUILD_DOCUMENTATION "Build doxygen auto documentation" ${KNP_BUILD_AUTONOMOUS} "DOXYGEN_FOUND" OFF)

# Check for CUDA.
find_package(CUDAToolkit)

option(KNP_BUILD_CUDA "Enable CUDA backend build" ${CUDAToolkit_FOUND})
option(KNP_BUILD_EXAMPLES "Build usage examples" ${KNP_BUILD_AUTONOMOUS})
option(KNP_BUILD_TESTS "Build tests" ${KNP_BUILD_AUTONOMOUS})
option(KNP_ENABLE_AVX "Enable AVX and other CPU-specific extensions in the release build" ${KNP_ENABLE_AVX_DEFAULT})
option(KNP_ENABLE_COVERAGE "Enable coverage checking" OFF)
option(KNP_IPO_ENABLED "Enable interprocedural optimization" ON)
option(KNP_INSTALL "Enable Kaspersky Neuromorphic Platform installation" ON)
option(KNP_MAINTAINER_BUILD "Build for maintainer, but not for the development purposes" OFF)
cmake_dependent_option(KNP_PYTHON_FRAMEWORK_BUILD "Build Kaspersky Neuromorphic Platform Python framework" ON "KNP_PYTHON_FRAMEWORK_BUILD_DEFAULT" OFF)
cmake_dependent_option(KNP_PYTHON_BUILD_WHEEL "Build WHL package for the Python framework" ${KNP_MAINTAINER_BUILD} "KNP_PYTHON_FRAMEWORK_BUILD" OFF)

mark_as_advanced(KNP_MAINTAINER_BUILD)
mark_as_advanced(KNP_ENABLE_COVERAGE)
mark_as_advanced(KNP_ENABLE_AVX)
mark_as_advanced(KNP_IPO_ENABLED)

message(STATUS "KNP_BUILD_CUDA = ${KNP_BUILD_CUDA}")
message(STATUS "KNP_BUILD_DOCUMENTATION = ${KNP_BUILD_DOCUMENTATION}")
message(STATUS "KNP_BUILD_EXAMPLES = ${KNP_BUILD_EXAMPLES}")
message(STATUS "KNP_BUILD_TESTS = ${KNP_BUILD_TESTS}")
message(STATUS "KNP_ENABLE_AVX = ${KNP_ENABLE_AVX}")
message(STATUS "KNP_ENABLE_COVERAGE = ${KNP_ENABLE_COVERAGE}")
message(STATUS "KNP_IPO_ENABLED = ${KNP_IPO_ENABLED}")
message(STATUS "KNP_INSTALL = ${KNP_INSTALL}")
message(STATUS "KNP_MAINTAINER_BUILD = ${KNP_MAINTAINER_BUILD}")
message(STATUS "KNP_PYTHON_FRAMEWORK_BUILD = ${KNP_PYTHON_FRAMEWORK_BUILD}")
message(STATUS "KNP_PYTHON_BUILD_WHEEL = ${KNP_PYTHON_BUILD_WHEEL}")

if (KNP_ENABLE_AVX)
    message(STATUS "AVX enabled.")
    message(STATUS "Setting release optimization flags...")
    # -pipe
    set(CMAKE_C_FLAGS_RELEASE "-march=native -mtune=native -mno-vzeroupper -Ofast -funroll-loops -fomit-frame-pointer -finline")

    # if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    #    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fopt-info-vec-optimized")
    # endif()

    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
    add_compile_options($<$<COMPILE_LANGUAGE:CXX,C>:-include>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX,C>:x86intrin.h>)
    # -include bits/stdc++.h
endif()

# TODO: fix it!
# An error has occurred: FatalError: git failed. Is it installed, and are you in a Git repository directory?
# Check the log at /root/.cache/pre-commit/pre-commit.log
#if (NOT KNP_MAINTAINER_BUILD AND KNP_BUILD_AUTONOMOUS)
#    execute_process(COMMAND pre-commit install
#                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
#    if(ret AND NOT ret EQUAL 0)
#        message(FATAL_ERROR "Cannot install pre-commit hooks.")
#    endif()
#    execute_process(COMMAND pre-commit install -t commit-msg
#                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
#endif()

if (KNP_BUILD_TESTS)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=0)
endif()


#
# Common parameters.
#

set(ENABLE_FAST_MATH ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Required for linters.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(knp-functions)
include(print-variables)

#
# Build third-party.
#

if (KNP_BUILD_AUTONOMOUS)
    if (NOT CPM_SOURCE_CACHE_DEFAULT)
        set(CPM_SOURCE_CACHE_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/third-party")
    endif()

    if (NOT CPM_SOURCE_CACHE)
        message(STATUS "Autonomous build. Setting source cached directory to \"${CPM_SOURCE_CACHE_DEFAULT}\".")
        set(CPM_SOURCE_CACHE "${CPM_SOURCE_CACHE_DEFAULT}")
    endif()
endif()

# Make cache variables for installation destinations.
include(GNUInstallDirs)

if (MSVC AND CMAKE_GENERATOR STREQUAL "Ninja")
    set(ZMQ_ENABLE_PRECOMPS OFF)
else()
    set(ZMQ_ENABLE_PRECOMPS ON)
endif()

add_third_party("gh:gabime/spdlog@1.13.0"
                OPTIONS
                    "SPDLOG_BUILD_PIC ON"
                    "SPDLOG_BUILD_SHARED OFF")

if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

add_third_party(NAME "libzmq"
                GITHUB_REPOSITORY "zeromq/libzmq"
                VERSION "4.3.5"
                FIND_PACKAGE_ARGUMENTS "NAMES ZeroMQ libzmq zmq zmq5 libzmq5"
                OPTIONS
                    "WITH_DOCS OFF"
                    "BUILD_TESTS OFF"
                    "ENABLE_CPACK OFF"
                    "ENABLE_PRECOMPILED ${ZMQ_ENABLE_PRECOMPS}")

add_third_party("gh:google/flatbuffers@24.3.25"
                OPTIONS
                    "FLATBUFFERS_BUILD_TESTS OFF"
                    "FLATBUFFERS_BUILD_CPP17 ON"
                    "FLATBUFFERS_INSTALL OFF")

macro(install)
    # Special hack needs to prevent redundant packages generation by third-party.
    if (CMAKE_CURRENT_SOURCE_DIR MATCHES "${CPM_SOURCE_CACHE}/.*" OR CMAKE_CURRENT_SOURCE_DIR MATCHES "${CPM_SOURCE_CACHE_DEFAULT}/.*")
        message(STATUS "Third-party installation will be skipped: ${CMAKE_CURRENT_SOURCE_DIR}")
        return()
    endif()
    _install(${ARGV})
endmacro()

#
# LibHDF5 and its dependencies.
#

#add_third_party("https://gitlab.dkrz.de/k202009/libaec.git@1.1.3")
#add_third_party("gh:erdc/szip")
add_third_party("gh:HDFGroup/hdf5#hdf5_1.14.6"
			    USE_LOCAL_PACKAGES NO
                GIT_SHALLOW ON
                FIND_PACKAGE_ARGUMENTS "NAMES hdf5 libhdf5 hdf5-static"
                OPTIONS
                    "HDF5_EXTERNALLY_CONFIGURED ON"
                    "HDF5_BUILD_DOC OFF"
                    "HDF5_NO_PACKAGES ON"
                    "H5_ENABLE_STATIC_LIB ON"
                    "HDF5_BUILD_SHARED_LIBS ON"
                    "HDF5_BUILD_TOOLS OFF"
                    "HDF5_BUILD_UTILS OFF"
                    "HDF5_ENABLE_Z_LIB_SUPPORT OFF"
                    "HDF5_USE_ZLIB_NG ON"
                    "HDF5_USE_ZLIB_STATIC ON"
                    "HDF5_VOL_ALLOW_EXTERNAL YES")
#                    "ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/zlib-ng-build")  # "HDF5_ALLOW_EXTERNAL_SUPPORT GIT" SZIP_USE_EXTERNAL

include_directories("${CPM_PACKAGE_hdf5_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/_deps/hdf5-build/src") # $<BUILD_INTERFACE:${CPM_PACKAGE_hdf5_SOURCE_DIR}>

add_third_party("gh:highfive-devs/highfive@3.1.0"
                GIT_SHALLOW ON
                OPTIONS
                    "HIGHFIVE_FIND_HDF5 OFF"
                    "HIGHFIVE_EXAMPLES OFF"
                    "HIGHFIVE_BUILD_DOCS OFF"
                    "HIGHFIVE_USE_BOOST OFF"
                    "HIGHFIVE_UNIT_TESTS OFF"
                    "HDF5_C_LIBRARIES ${HDF5_LIBRARIES_TO_EXPORT}")

# End of LibHDF5.

add_third_party("gh:p-ranav/csv2@0.1"
                OPTIONS
                    "CSV2_BUILD_TESTS OFF"
                    "CSV2_TEST OFF"
                    "CSV2_SAMPLES OFF")

if ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86.*)|(X86.*)|(amd64)|(AMD64)" AND NOT APPLE)
    add_third_party("gh:intel/pcm"
                    GIT_TAG 2f9b0c2
                    OPTIONS
                        "NO_STATIC_MSVC_RUNTIME_LIBRARY ON"
                        "NO_ASAN ON")
    include_directories("${CPM_PACKAGE_pcm_SOURCE_DIR}/src")
endif()

add_third_party("gh:zeromq/cppzmq@4.11.0"
                OPTIONS
                    "CPPZMQ_BUILD_TESTS OFF")

add_third_party("gh:Tencent/rapidjson#ebd87cb"
                OPTIONS
                    "RAPIDJSON_BUILD_DOC OFF"
                    "RAPIDJSON_BUILD_ASAN OFF"
                    "RAPIDJSON_BUILD_UBSAN OFF"
                    "RAPIDJSON_BUILD_CXX17 ON"
                    "RAPIDJSON_BUILD_EXAMPLES OFF"
                    "RAPIDJSON_BUILD_TESTS OFF"
                    "RAPIDJSON_HAS_STDSTRING ON")

include_directories("${CPM_PACKAGE_rapidjson_SOURCE_DIR}/include")

if (KNP_BUILD_TESTS)
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=0)
    add_third_party("gh:google/googletest@1.14.0"
                    OPTIONS
                        "INSTALL_GMOCK OFF"
                        "INSTALL_GTEST OFF")
endif()

#if (CPM_SOURCE_CACHE)
#    file(GLOB THIRD_PARTY_INCLUDES LIST_DIRECTORIES true "${CPM_SOURCE_CACHE}/**/include")
#    include_directories("${CPM_SOURCE_CACHE}" ${THIRD_PARTY_INCLUDES})
#endif()

if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
endif()


#
# Build Kaspersky Neuromorphic Platform.
#

set(CMAKE_TIDY_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/knp/.clang-tidy" CACHE PATH  "Clang-tidy config file")

add_subdirectory("knp")

if (KNP_BUILD_DOCUMENTATION)
    add_subdirectory(doc)
endif()

if (KNP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

#
# Packaging.
#

set(CPACK_PACKAGE_NAME "${PROJECT_NAME}" CACHE STRING "The resulting package name")

#SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}.${CMAKE_SYSTEM_PROCESSOR}")
#set(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_ICON "packaging/neuromorphic-platform.svg")

set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})

set(CPACK_PACKAGE_VENDOR "AO Kaspersky Lab")
set(CPACK_PACKAGE_CONTACT "support@kaspersky.com")

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${CMAKE_PROJECT_DESCRIPTION}"
        CACHE STRING "Package description for the package metadata")
set(CPACK_PACKAGE_DESCRIPTION "Kaspersky Neuromorphic Platform - spiking networks simulator")

set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/_packages")

# https://unix.stackexchange.com/a/11552/254512
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

# On Linux, grouping must be ignored: one component corresponds to one package.
set(CPACK_COMPONENTS_GROUPING IGNORE)


#
# Deb package.
#

set(CPACK_DEB_PACKAGE_COMPONENT ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
#set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Artiom N.")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION}")

set(CPACK_DEBIAN_PACKAGE_SECTION "science")

set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS OFF)


#
# RPM package.
#

set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_RELOCATABLE  ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

set(CPACK_RPM_PACKAGE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION}")
set(CPACK_RPM_PACKAGE_GROUP "science")

include (CPack)
