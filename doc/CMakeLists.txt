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
project(knp-documentation
        VERSION "${KNP_VERSION}"
        DESCRIPTION "Documentation: user guide and reference")

include(GNUInstallDirs)
include(knp-functions)

knp_get_component_name("${PROJECT_NAME}" COMPONENT_NAME)

install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/html"
#        "${CMAKE_CURRENT_SOURCE_DIR}/pdf"
        COMPONENT "${COMPONENT_NAME}"
        TYPE DOC)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/html/index.html" FILE_CONTENTS)
string(REPLACE "../../build/doc_doxygen/html/index.html" "reference/index.html" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/index.html" "${FILE_CONTENTS}")

if (KNP_INSTALL)
    set(COMPONENT_NAME "documentation")

    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/index.html"
            COMPONENT "${COMPONENT_NAME}"
            TYPE DOC
            RENAME "html/index.html")

    knp_packaging_set_parameters("${COMPONENT_NAME}" "${PROJECT_NAME}"
            DESCRIPTION "${PROJECT_DESCRIPTION}"
            INTERNAL_DEPENDS cpp-framework-dev
            PROVIDES "knp-documentation")

    knp_get_component_var_name("${COMPONENT_NAME}" COMPONENT_VAR)

    set(CPACK_DEBIAN_${COMPONENT_VAR}_PACKAGE_ARCHITECTURE "all" CACHE STRING "${PROJECT_NAME} architecture")
    set(CPACK_DEBIAN_${COMPONENT_VAR}_PACKAGE_SHLIBDEPS OFF CACHE BOOL "")
endif()
