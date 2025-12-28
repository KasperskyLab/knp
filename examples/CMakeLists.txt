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

include(GNUInstallDirs)
include(knp-functions)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(save-load-networks)
add_subdirectory(simple-network)
add_subdirectory(mnist-client)
add_subdirectory(mnist-learn)

if (KNP_INSTALL)
    install(DIRECTORY "."
            COMPONENT "examples"
            DESTINATION "src/knp/examples")

    file(GLOB EXAMPLES_DIRS LIST_DIRECTORIES true "*")
    set(INSTALL_TARGETS "")

    foreach(_dir ${EXAMPLES_DIRS})
        if (IS_DIRECTORY "${_dir}")
            get_filename_component(DIR_NAME "${_dir}" NAME)
            get_directory_property(DIR_TARGETS DIRECTORY "${_dir}" BUILDSYSTEM_TARGETS)
            foreach(INSTALL_TARGET ${DIR_TARGETS})
                list(APPEND INSTALL_TARGETS "${INSTALL_TARGET}")
            endforeach()
        endif()
    endforeach()

    install(TARGETS ${INSTALL_TARGETS}
            COMPONENT "examples")

    knp_packaging_set_parameters(examples examples
            DESCRIPTION "Kaspersky Neuromorphic Platform examples"
            DEPENDS cpp-framework-dev)
endif()
