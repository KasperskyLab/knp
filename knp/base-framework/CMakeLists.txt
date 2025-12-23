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

project(knp-base-framework VERSION "${KNP_VERSION}" LANGUAGES C CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform C++ framework library"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

set(${PROJECT_NAME}_PUBLIC_INCLUDE_DIR "knp/framework")

find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS filesystem REQUIRED)

include(GNUInstallDirs)
include(knp-functions)

file(GLOB_RECURSE ${PROJECT_NAME}_headers include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/*.h)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(ADD_LIBS "supc++")
endif()

knp_add_library("${PROJECT_NAME}-core"
    BOTH
    impl/backend_loader.cpp
    impl/storage/native/data_storage_common.cpp
    impl/storage/native/data_storage_json.cpp
    impl/storage/native/data_storage_hdf5.cpp
    impl/network.cpp
    impl/model.cpp
    impl/model_executor.cpp
    impl/model_loader.cpp
    impl/message_handlers.cpp
    impl/input_converter.cpp
    impl/output_channel.cpp
    impl/synchronization.cpp
    impl/monitoring/model.cpp
    impl/projection/wta.cpp
    impl/sonata/save_network.cpp
    impl/sonata/load_network.cpp
    impl/sonata/csv_content.cpp
    impl/sonata/types/blifat_neuron.cpp
    impl/sonata/types/delta_synapse.cpp
    impl/sonata/types/resource_blifat_neuron.cpp
    impl/sonata/types/altai_lif_neuron.cpp
    impl/sonata/types/resource_delta_synapse.cpp
    impl/sonata/types/additive_delta_synapse.cpp
    impl/data_processing/classification/dataset.cpp
    impl/data_processing/classification/image.cpp
    impl/inference_evaluation/perfomance_metrics.cpp
    impl/inference_evaluation/classification/processor.cpp
    impl/observer.cpp
    ${${PROJECT_NAME}_headers}
    ALIAS KNP::BaseFramework::Core
    LINK_PRIVATE
        spdlog::spdlog_header_only Boost::headers Boost::filesystem HighFive hdf5-static csv2
        # Hack to build with CLang.
        ${ADD_LIBS}
    LINK_PUBLIC
        KNP::Core KNP::Neuron::Traits KNP::Synapse::Traits
    #PRECOMP KNP::Core
)

set_target_properties(
    "${PROJECT_NAME}-core"
    PROPERTIES
    VERSION "${CMAKE_PROJECT_VERSION}"
    SOVERSION "${CMAKE_PROJECT_VERSION_MAJOR}")

if (KNP_INSTALL)
    set(COMPONENT_NAME "cpp-framework")

    configure_package_config_file(
            "${CMAKE_CURRENT_LIST_DIR}/packaging/${PROJECT_NAME}Config.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/packaging/${PROJECT_NAME}Config.cmake"
            INSTALL_DESTINATION
            "${CMAKE_INSTALL_LIBDIR}/cmake/")

    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/packaging/${PROJECT_NAME}ConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion)

    install(TARGETS "${PROJECT_NAME}-core"
            EXPORT "${PROJECT_NAME}-core"
            LIBRARY
            COMPONENT ${COMPONENT_NAME})

    install(FILES
            "${CMAKE_SOURCE_DIR}/packaging/LICENSE.txt"
            "${CMAKE_SOURCE_DIR}/packaging/NOTICE.txt"
            "${CMAKE_SOURCE_DIR}/README.md"
            COMPONENT ${COMPONENT_NAME}
            TYPE DOC)

    install(TARGETS "${PROJECT_NAME}-core_static"
            EXPORT "${PROJECT_NAME}-core_static"
            COMPONENT ${COMPONENT_NAME}-dev
            ARCHIVE)

    install(DIRECTORY "include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}"
            COMPONENT ${COMPONENT_NAME}-dev
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/knp"
            FILES_MATCHING PATTERN "*.h")

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/packaging/${PROJECT_NAME}Config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/packaging/${PROJECT_NAME}ConfigVersion.cmake"
            COMPONENT ${COMPONENT_NAME}-dev
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

    knp_packaging_set_parameters(${COMPONENT_NAME} "${PROJECT_NAME}"
        DESCRIPTION "${PROJECT_DESCRIPTION}"
        # DEPENDS "backend" "libboost-all (= ${Boost_VERSION})" "libhdf5" "libzmq5"
    )

    knp_packaging_set_parameters(${COMPONENT_NAME}-dev "${PROJECT_NAME}"
            DESCRIPTION "${PROJECT_DESCRIPTION} development files: C++ headers, CMake modules, static libraries"
            DEPENDS "${COMPONENT_NAME}" "libboost-all-dev (= ${Boost_VERSION})"
            RECOMMENDS "cppzmq-dev (>= 4.9.0-1)" "cppzmq (= 4.9.0-1)")
endif()
