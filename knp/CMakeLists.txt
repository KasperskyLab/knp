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

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Required for linters.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(knp VERSION "${KNP_VERSION}" LANGUAGES C CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

include(pvs-studio)

# TODO: Enable the latest Boost versions: this doesn't work for DLLs on the old versions.
# set(Boost_USE_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (KNP_ENABLE_COVERAGE)
    include(codecoverage)
    append_coverage_compiler_flags()
endif()

add_subdirectory(meta-library)
add_subdirectory(neuron-traits-library)
add_subdirectory(synapse-traits-library)
add_subdirectory(core-library)
add_subdirectory(backends)
add_subdirectory(base-framework)
add_subdirectory(python-framework)
add_subdirectory(autodoc)
add_subdirectory(tests)

file(GLOB PVS_DIRS LIST_DIRECTORIES true "*")
file(GLOB dirs LIST_DIRECTORIES true "backends/cpu/*")

set(PVS_DIRS "${PVS_DIRS};${dirs}")

if (WIN32)
    find_program(PVS_BINARY PVS-Studio_Cmd.exe HINTS "$ENV{ProgramFiles}\\PVS-Studio" "$ENV{ProgramFiles\(x86\)}\\PVS-Studio" NO_CACHE)
else()
    find_program(PVS_BINARY pvs-studio-analyzer NO_CACHE)
endif()

if (PVS_BINARY)
    set(PVS_TARGETS "")
    set(PVS_SKIP_LIST ".*autodoc|GENERATE_.*|.*-copy-.*|.*memcheck|.*_static")

    foreach(_dir ${PVS_DIRS})
        if (IS_DIRECTORY "${_dir}")
            get_filename_component(DIR_NAME "${_dir}" NAME)
            get_directory_property(DIR_TARGETS DIRECTORY "${_dir}" BUILDSYSTEM_TARGETS)
            foreach(PVS_TARGET ${DIR_TARGETS})
                string(REGEX MATCH "${PVS_SKIP_LIST}" PVS_SKIP ${PVS_TARGET})
                if (PVS_SKIP)
                    continue()
                endif()
                list(APPEND PVS_TARGETS "${PVS_TARGET}")
            endforeach()
        endif()
    endforeach()

    pvs_studio_add_target(TARGET "pvs-analyze"
                          OUTPUT FORMAT tasklist
                          ANALYZE ${PVS_TARGETS}
                          MODE GA:1,2
                          ARGS -e "${CMAKE_SOURCE_DIR}/third-party"
                          ARGS -e "/usr/local")
else()
    message(WARNING "PVS analyzer was not found, static analysis with PVS Studio disabled.")
endif()
