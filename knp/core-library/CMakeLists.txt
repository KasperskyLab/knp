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

# Required it for the linters.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

project(knp-core VERSION "${KNP_VERSION}" LANGUAGES C CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform core library"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

if(NOT TARGET Boost::headers)
    find_package(Boost ${KNP_BOOST_MIN_VERSION} REQUIRED)
endif()

if(NOT TARGET cppzmq)
    find_package(cppzmq REQUIRED)
endif()

if (MSVC)
    set(CPP_ZMQ cppzmq-static)
else()
    set(CPP_ZMQ cppzmq)
endif()

include(GNUInstallDirs)
# Need it for installation.
include(CMakePackageConfigHelpers)
include(clang-tidy)
include(knp-functions)
include(third-party)

set(${PROJECT_NAME}_PUBLIC_INCLUDE_DIR "knp/core")

file(GLOB_RECURSE ${PROJECT_NAME}_headers include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/*.h)

set(${PROJECT_NAME}_FB_SOURCES
    impl/messaging/fbs/message_envelope.fbs
    impl/messaging/fbs/message_header.fbs
    impl/messaging/fbs/synapse_traits.fbs
    impl/messaging/fbs/spike_message.fbs
    impl/messaging/fbs/synaptic_impact_message.fbs)

#
# Build.
#

flatbuffers_generate_headers(
    TARGET "${PROJECT_NAME}_messaging"
    INCLUDE_PREFIX "knp_gen_headers"
    SCHEMAS ${${PROJECT_NAME}_FB_SOURCES}
    # BINARY_SCHEMAS_DIR "${CMAKE_BINARY_DIR}/gen_includes"
    # FLAGS --gen-object-api
)

knp_add_library("${PROJECT_NAME}"
    STATIC
    impl/backend.cpp
    impl/device.cpp
    impl/population.cpp
    impl/uid.cpp
    impl/projection.cpp
    impl/message_bus.cpp
    impl/message_endpoint.cpp
    impl/message_bus_zmq_impl/message_bus_zmq_impl.h
    impl/message_bus_zmq_impl/message_endpoint_zmq_impl.h
    impl/message_bus_zmq_impl/message_bus_zmq_impl.cpp
    impl/message_bus_zmq_impl/message_endpoint_zmq_impl.cpp
    impl/message_bus_cpu_impl/message_bus_cpu_impl.cpp
    impl/message_bus_cpu_impl/message_bus_cpu_impl.h
    impl/message_bus_cpu_impl/message_endpoint_cpu_impl.h
    impl/message_bus_impl.h
    impl/message_header.cpp
    impl/messaging/message_envelope.cpp
    impl/messaging/uid_marshal.h
    impl/messaging/spike_message_impl.h
    impl/messaging/spike_message.cpp
    impl/messaging/synaptic_impact_message_impl.h
    impl/messaging/synaptic_impact_message.cpp
    impl/subscription.cpp

    ${${PROJECT_NAME}_headers}
    # PRECOMP impl/common_precomp.h
    LINK_PRIVATE
        Boost::headers spdlog::spdlog_header_only ${CPP_ZMQ} flatbuffers "${PROJECT_NAME}_messaging"
    LINK_PUBLIC
        # This is used in the library for message parameters.
        KNP::Neuron::Traits KNP::Synapse::Traits
    ALIAS KNP::Core
)

source_group(source REGULAR_EXPRESSION "impl/.*")

target_include_directories("${PROJECT_NAME}" PRIVATE ${Boost_INCLUDE_DIRS} "impl")

# Flatbuffer headers must be generated before core compilation starts.
add_dependencies("${PROJECT_NAME}" "GENERATE_${PROJECT_NAME}_messaging" "${PROJECT_NAME}_messaging")

# add_clang_tidy("${PROJECT_NAME}" CONFIG_FILE_PATH "${CMAKE_TIDY_CONFIG}" EXTRA_ARGS "--use-color")


#
# Installation.
#

message(STATUS "Configuring installation...")

set(PACKAGE_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR})

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/version.h")


if (KNP_INSTALL)
    set(COMPONENT_NAME "cpp-framework")

    install(TARGETS "${PROJECT_NAME}"
            COMPONENT "${COMPONENT_NAME}-dev"
            EXPORT "${COMPONENT_NAME}-dev_targets"
            ARCHIVE)

#    install(TARGETS "${PROJECT_NAME}_messaging"
#            COMPONENT "${COMPONENT_NAME}-dev"
#            EXPORT "${COMPONENT_NAME}-dev_targets")

    install(DIRECTORY "include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}"
            COMPONENT "${COMPONENT_NAME}-dev"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/knp"
            FILES_MATCHING PATTERN "*.h")

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/version.h"
            COMPONENT "${COMPONENT_NAME}-dev"
            DESTINATION "include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}")
endif()
