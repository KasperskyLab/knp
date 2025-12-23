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

project(knp_python_framework VERSION "${KNP_VERSION}" LANGUAGES C CXX  # PYTHON
        DESCRIPTION "Kaspersky Neuromorphic Platform Python framework library"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

if (NOT KNP_PYTHON_FRAMEWORK_BUILD)
    message(STATUS "Python framework building disabled.")
    return()
endif()

message(STATUS "Python framework building enabled.")

if (Boost_ROOT)
    set(Boost_USE_STATIC_LIBS ON)
#    add_compile_definitions(-DBOOST_PYTHON_STATIC_LIB -DPy_NO_ENABLE_SHARED)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python version: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")

if (WIN32)
    find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR} CONFIG)
else()
    find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS python CONFIG)
endif()

if (NOT Boost_PYTHON_FOUND)
    message(WARNING "Correct Boost::Python was not found, trying heuristic search...")
    file(GLOB _KNP_PY_VER "${BOOST_LIBRARYDIR}/*boost_python*.*")

    if (_KNP_PY_VER)
        list(GET _KNP_PY_VER 0 _KNP_PY_VER)
        string(REGEX REPLACE ".+python([0-9]+).*[\n\r]*" "\\1" _KNP_PY_VER "${_KNP_PY_VER}")
    else()
        message(FATAL_ERROR "\"${BOOST_LIBRARYDIR}/*boost_python*.*\" was not found!")
    endif()

    message(NOTICE "Trying to find Boost::Python library version: python${_KNP_PY_VER}")
    find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS python${_KNP_PY_VER} REQUIRED)
endif()

include(knp-functions)
include(file-functions)

set(KNP_PYTHON_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}")

knp_add_python_module(core
        LINK_LIBRARIES KNP::Core KNP::BaseFramework::Core
        PY_VER ${_KNP_PY_VER})
knp_add_python_module(neuron_traits
        LINK_LIBRARIES KNP::Neuron::Traits
        PY_VER ${_KNP_PY_VER})
knp_add_python_module(synapse_traits
        LINK_LIBRARIES KNP::Synapse::Traits
        PY_VER ${_KNP_PY_VER})
knp_add_python_module(base_framework
        LINK_LIBRARIES KNP::BaseFramework::Core
        PY_VER ${_KNP_PY_VER})

add_custom_target("${PROJECT_NAME}-copy-python-tests" ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/tests" "${KNP_PYTHON_OUTPUT_DIRECTORY}/tests")
add_custom_target("${PROJECT_NAME}-copy-common-files" ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/packaging/common" "${KNP_PYTHON_OUTPUT_DIRECTORY}")
add_custom_target("${PROJECT_NAME}-copy-common-python-files" ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/packaging/python" "${KNP_PYTHON_OUTPUT_DIRECTORY}/knp")
add_custom_target("${PROJECT_NAME}-copy-common-project-files" ALL
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/../../AUTHORS"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE.txt"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../README.md"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION"
            "${KNP_PYTHON_OUTPUT_DIRECTORY}")

find_program(MYPY_STUBGEN stubgen REQUIRED)
get_property(KNP_PYTHON_TARGETS DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)

add_custom_target("${PROJECT_NAME}-generate-python-interface-files" ALL
        WORKING_DIRECTORY "${KNP_PYTHON_OUTPUT_DIRECTORY}"
        COMMAND "${MYPY_STUBGEN}" -p "knp" -o "." --include-private
        DEPENDS ${KNP_PYTHON_TARGETS})

get_property(KNP_PYTHON_TARGETS DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)

if(KNP_BUILD_TESTS)
    add_test(NAME Pytest
            WORKING_DIRECTORY "${KNP_PYTHON_OUTPUT_DIRECTORY}"
            COMMAND "${Python_EXECUTABLE}" -m pytest)
endif()

if (KNP_INSTALL)
    set(COMPONENT_NAME "python3-framework")

    install(DIRECTORY
            "${KNP_PYTHON_OUTPUT_DIRECTORY}/knp"
            COMPONENT "${COMPONENT_NAME}"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/python3"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "__init__.pyi" EXCLUDE
            PATTERN "*.egg-info" EXCLUDE
            PATTERN "*.gcno" EXCLUDE)

    knp_packaging_set_parameters("${COMPONENT_NAME}" "${PROJECT_NAME}"
            DESCRIPTION "${PROJECT_DESCRIPTION}"
            DEPENDS "cpp-framework" "python3 (>= 3.10)")
endif()

#
# PIP packaging.
#

if(KNP_PYTHON_BUILD_WHEEL)
    add_custom_command(
        OUTPUT "${KNP_PYTHON_OUTPUT_DIRECTORY}/dist/knp-${PROJECT_VERSION}-py3-none-any.whl"
        COMMAND "${Python3_EXECUTABLE}" -m build --wheel
        WORKING_DIRECTORY "${KNP_PYTHON_OUTPUT_DIRECTORY}"
    )

    # TODO: Automatically get all targets in CMakeLists.txt.
    add_custom_target(
        "${PROJECT_NAME}-pip-build-package" ALL
        DEPENDS
            "${PROJECT_NAME}_base_framework"
            "${PROJECT_NAME}_core"
            "${PROJECT_NAME}_neuron_traits"
            "${PROJECT_NAME}_synapse_traits"
            "${PROJECT_NAME}-copy-python-tests"
            "${PROJECT_NAME}-copy-common-files"
            "${PROJECT_NAME}-copy-common-python-files"
            "${PROJECT_NAME}-copy-common-project-files"
            "${KNP_PYTHON_OUTPUT_DIRECTORY}/dist/knp-${PROJECT_VERSION}-py3-none-any.whl"
    )
endif()