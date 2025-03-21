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

#
# Kaspersky Neuromorphic Platform build functions. Artiom N.(cl)2023.
#

include_guard(GLOBAL)

include(CheckIPOSupported)


function(knp_get_hdf5_target target_name)
    if (TARGET hdf5-static)
        message(STATUS "Using static HDF5 library...")
        set(${target_name} hdf5-static PARENT_SCOPE)
    elseif (TARGET HDF5::HDF5)
        message(STATUS "Using dynamic HDF5 library...")
        set(${target_name} HDF5::HDF5 PARENT_SCOPE)
    else()
        message(FATAL_ERROR "HDF5 library was not found.")
    endif()
endfunction()


function(knp_set_target_parameters target visibility)
    if (${visibility} STREQUAL "INTERFACE")
        set(pub_visibility "INTERFACE")
    else()
        set(pub_visibility "PUBLIC")
        if(KNP_IPO_ENABLED)
            check_ipo_supported(RESULT _ipo_result OUTPUT _ipo_output)
            if (_ipo_result)
                set_target_properties("${target}" PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
            else()
                message(WARNING "IPO is not supported: ${_ipo_output}.")
            endif()
        endif()
    endif()

    # if (MSVC)  # AND ($<COMPILE_LANGUAGE> STREQUAL CXX OR $<COMPILE_LANGUAGE> STREQUAL C))
    #    set_property(TARGET "${target}" PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # endif()

    target_compile_options("${target}" ${visibility} $<$<COMPILE_LANG_AND_ID:C,Clang>:-Wdocumentation>)
    target_compile_options("${target}" ${visibility} $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wdocumentation>)

    # SDL requirements.
    target_compile_options("${target}" ${visibility} $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall -fstack-protector-all -Wformat -Wformat-security>)
    target_compile_options("${target}" ${visibility} $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -fstack-protector-all -Wformat -Wformat-security>)

    target_compile_features("${target}" ${pub_visibility} cxx_std_17)

    target_compile_definitions("${target}" ${visibility}
            #                                $<$<CONFIG:Debug>:_FORTIFY_SOURCE=2>
            # $<$<CONFIG:Release>:_FORTIFY_SOURCE=1>
            )

    target_compile_definitions("${target}" ${visibility} $<$<COMPILE_LANG_AND_ID:C,MSVC>:NOMINMAX>)
    target_compile_definitions("${target}" ${visibility} $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:NOMINMAX>)

    # Sanitizer (dynamic analysis).
    # Sanitizers don't work under TFS. Temporarily disabled.

    # set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address")
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address -fsanitize-recover=address")
    if (UNIX AND NOT APPLE)
        #        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro,-z,now")
    endif()

    target_include_directories("${target}" ${pub_visibility}
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
endfunction()


function (_knp_add_library lib_name lib_type)
    set(options "NO_PREFIX")
    set(one_value_args "ALIAS;LIB_PREFIX;PRECOMP")
    set(multi_value_args "LINK_PRIVATE;LINK_PUBLIC")

    cmake_parse_arguments(PARSED_ARGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    set(${lib_name}_source ${PARSED_ARGS_UNPARSED_ARGUMENTS})

    if ("${lib_type}" STREQUAL "PY_MODULE")
        python3_add_library("${lib_name}" MODULE ${${lib_name}_source})
    else()
        add_library("${lib_name}" ${lib_type} ${${lib_name}_source})
    endif()

    if (PARSED_ARGS_ALIAS)
        add_library(${PARSED_ARGS_ALIAS} ALIAS "${lib_name}")
    endif()

    if (PARSED_ARGS_PRECOMP)
        string(REGEX REPLACE "[:-]" "_" _PRECOMP_NAME ${PARSED_ARGS_PRECOMP})
        if (_KNP_PRECOMP_${_PRECOMP_NAME}_)
            # Precomp already exists.
            target_precompile_headers("${lib_name}" REUSE_FROM "${_KNP_PRECOMP_${_PRECOMP_NAME}_}")
        else()
            # New precomp.
            # PARSED_ARGS_PRECOMP may be a path: it must not be changed.
            target_precompile_headers("${lib_name}" PRIVATE "${PARSED_ARGS_PRECOMP}")
            set(_KNP_PRECOMP_${_PRECOMP_NAME}_ "${lib_name}" CACHE STRING "Precomp for ${lib_name}")
            if(PARSED_ARGS_ALIAS)
                string(REGEX REPLACE "[:-]" "_" _PRECOMP_NAME ${PARSED_ARGS_ALIAS})
                set(_KNP_PRECOMP_${_PRECOMP_NAME}_ "${lib_name}" CACHE STRING "Precomp for ${lib_name} alias ${PARSED_ARGS_ALIAS}")
            endif()
        endif()
    endif()

    if (PARSED_ARGS_LINK_PUBLIC)
        target_link_libraries("${lib_name}" PUBLIC ${PARSED_ARGS_LINK_PUBLIC})
    endif()

    if (PARSED_ARGS_LINK_PRIVATE)
        if ("${lib_type}" STREQUAL "STATIC")
            target_link_libraries("${lib_name}" PUBLIC ${PARSED_ARGS_LINK_PRIVATE})
        else()
            target_link_libraries("${lib_name}" PRIVATE ${PARSED_ARGS_LINK_PRIVATE})
        endif()
    endif()

    if ("${lib_type}" STREQUAL "INTERFACE")
        knp_set_target_parameters("${lib_name}" INTERFACE)
        set(_visibility "INTERFACE")
    else()
        knp_set_target_parameters("${lib_name}" PRIVATE)
        set(_visibility "PRIVATE")

        target_compile_definitions("${lib_name}" ${_visibility} "KNP_LIBRARY_NAME=${lib_name}")

        if (PARSED_ARGS_LIB_PREFIX)
            target_compile_definitions("${lib_name}" ${_visibility} "KNP_LIBRARY_NAME_PREFIX=${PARSED_ARGS_LIB_PREFIX}")
            target_compile_definitions("${lib_name}" ${_visibility} "KNP_FULL_LIBRARY_NAME=${PARSED_ARGS_LIB_PREFIX}${lib_name}")
            set_target_properties("${lib_name}" PROPERTIES PREFIX "${PARSED_ARGS_LIB_PREFIX}")
        elseif(NOT PARSED_ARGS_NO_PREFIX)
            target_compile_definitions("${lib_name}" ${_visibility} "KNP_LIBRARY_NAME_PREFIX=${CMAKE_STATIC_LIBRARY_PREFIX}")
            target_compile_definitions("${lib_name}" ${_visibility} "KNP_FULL_LIBRARY_NAME=${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}")
        endif()
    endif()
endfunction()


function (knp_add_library lib_name lib_type)
    string(TOUPPER "${lib_type}" lib_type)

    if (NOT lib_type OR lib_type STREQUAL "BOTH")
        _knp_add_library("${lib_name}" SHARED ${ARGN})
        if (ALIAS IN_LIST ARGN)
            # Replace alias for the static library.
            list(FIND ARGN ALIAS _index)
            math(EXPR _index "${_index} + 1")
            list(GET ARGN ${_index} _alias_value)
            list(INSERT ARGN ${_index} "${_alias_value}Static")
            math(EXPR _index "${_index} + 1")
            list(REMOVE_AT ARGN ${_index})
        endif()

        _knp_add_library("${lib_name}_static" STATIC ${ARGN})
        target_compile_definitions("${lib_name}" PRIVATE _KNP_BUILD_SHARED_LIBS)
        target_compile_definitions("${lib_name}_static" PRIVATE _KNP_INTERNAL)
        if (NOT WIN32)
            set_target_properties("${lib_name}_static" PROPERTIES OUTPUT_NAME "${lib_name}")
        endif()
    elseif (lib_type STREQUAL SHARED OR lib_type STREQUAL MODULE OR lib_type STREQUAL PY_MODULE)
        _knp_add_library("${lib_name}" ${lib_type} ${ARGN})
        target_compile_definitions("${lib_name}" PRIVATE _KNP_BUILD_SHARED_LIBS)
    elseif (lib_type STREQUAL STATIC OR lib_type STREQUAL INTERFACE)
        _knp_add_library("${lib_name}" ${lib_type} ${ARGN})
        # No need to set export definitions.
        return()
    else()
        message(FATAL_ERROR "Incorrect library build type: \"${lib_type}\". Use SHARED/PY_MODULE, STATIC, BOTH or INTERFACE.")
    endif()
    target_compile_definitions("${lib_name}" PRIVATE _KNP_INTERNAL)
endfunction()


# Function:                 knp_add_python_library
# Description:              Add Python module, using Boost::Python.
# Param name:               Module name.
# Param LINK_LIBRARIES:     Additional libraries to link.
# Requires:
#     find_package(Python COMPONENTS Interpreter Development REQUIRED)
#     find_package(Boost 1.66.0 COMPONENTS python REQUIRED)
function(knp_add_python_module name)
    cmake_parse_arguments(
            "PARSED_ARGS"
            ""
            "PY_VER"
            "LINK_LIBRARIES;CPP_SOURCE_DIRECTORY;OUTPUT_DIRECTORY"
            ${ARGN}
    )

    set(LIB_NAME "${PROJECT_NAME}_${name}")

    if (NOT PARSED_ARGS_CPP_SOURCE_DIRECTORY)
        set(PARSED_ARGS_CPP_SOURCE_DIRECTORY "cpp")
    endif()

    file(GLOB_RECURSE ${name}_MODULES_SOURCE CONFIGURE_DEPENDS
            "impl/${name}/${PARSED_ARGS_CPP_SOURCE_DIRECTORY}/*.h"
            "impl/${name}/${PARSED_ARGS_CPP_SOURCE_DIRECTORY}/*.cpp")

    knp_add_library(
            "${LIB_NAME}"
            PY_MODULE
            LIB_PREFIX "_"
            ${${name}_MODULES_SOURCE}
    )

    if (NOT PARSED_ARGS_OUTPUT_DIRECTORY)
        set(PARSED_ARGS_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/knp_python_framework/knp")
    endif()

    target_include_directories("${LIB_NAME}" PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries("${LIB_NAME}" PRIVATE Boost::headers Boost::python${PARSED_ARGS_PY_VER} ${PARSED_ARGS_LINK_LIBRARIES})

#    set_target_properties(${LIB_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
#    set_target_properties(${LIB_NAME} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

    set_target_properties("${LIB_NAME}" PROPERTIES LIBRARY_OUTPUT_DIRECTORY
            "$<BUILD_INTERFACE:${PARSED_ARGS_OUTPUT_DIRECTORY}/${name}>$<INSTALL_INTERFACE:LIBRARY_OUTPUT_DIRECTORY=${Python_SITEARCH}/knp/${name}>")

    target_compile_definitions("${LIB_NAME}" PRIVATE
            BOOST_PYTHON_STATIC_LIB
            Py_NO_ENABLE_SHARED
            CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON)

    add_custom_command(TARGET "${LIB_NAME}" POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/impl/${name}/python" "$<TARGET_FILE_DIR:${LIB_NAME}>")
endfunction()


function(knp_get_component_name project_name component_name)
    string(REGEX REPLACE "(knp-)([^ ]*)" "\\2" _component_name "${project_name}")
    set(${component_name}  ${_component_name} PARENT_SCOPE)
endfunction()


function(knp_get_component_var_name component_name component_var_name)
    string(TOUPPER "${component_name}" _component_var_name)
    set(${component_var_name} "${_component_var_name}" PARENT_SCOPE)
endfunction()


function(knp_packaging_set_parameters component_name project_name)
    cmake_parse_arguments(
            "PARSED_ARGS"
            ""
            "DESCRIPTION"
            "DEPENDS;PROVIDES;RECOMMENDS"
            ${ARGN}
    )

    knp_get_component_var_name("${component_name}" COMPONENT_VAR_NAME)

    set(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_FILE_NAME "knp_${component_name}_${CPACK_DEBIAN_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb"
            CACHE STRING "${project_name} package filename" FORCE)

    if (PARSED_ARGS_DESCRIPTION)
        set(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_DESCRIPTION "${PARSED_ARGS_DESCRIPTION}"
            CACHE STRING "${project_name} description" FORCE)
    else()
        unset(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_DESCRIPTION CACHE)
    endif()

    if (PARSED_ARGS_DEPENDS)
        string(REPLACE ";" ", " PARSED_ARGS_DEPENDS "${PARSED_ARGS_DEPENDS}")
        set(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_DEPE  NDS "${PARSED_ARGS_DEPENDS}" CACHE STRING "${project_name} dependencies" FORCE)
    else()
        unset(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_DEPENDS CACHE)
    endif()

    if (PARSED_ARGS_PROVIDES)
        set(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_PROVIDES "${PARSED_ARGS_PROVIDES}" CACHE STRING "${project_name} provides" FORCE)
    else()
        unset(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_PROVIDES CACHE)
    endif()

    if (PARSED_ARGS_RECOMMENDS)
        string(REPLACE ";" ", " PARSED_ARGS_RECOMMENDS "${PARSED_ARGS_RECOMMENDS}")
        set(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_RECOMMENDS "${PARSED_ARGS_RECOMMENDS}" CACHE STRING "${project_name} deb package recommendations" FORCE)
    else()
        unset(CPACK_COMPONENT_${COMPONENT_VAR_NAME}_RECOMMENDS CACHE)
    endif()
endfunction()
