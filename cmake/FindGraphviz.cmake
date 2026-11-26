# - Find Graphviz.

include(FindPackageHandleStandardArgs)

if ("Runtime" IN_LIST Graphviz_FIND_COMPONENTS)
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/GraphvizRuntime.txt" _graphviz_names)

    set(Graphviz_Runtime_FOUND TRUE)
    set(_graphviz_missing "")

    foreach(_name IN LISTS _graphviz_names)
        unset(_graphviz_file)

        if (WIN32)
            find_file(_graphviz_file NAMES "${_name}"
                HINTS "${Graphviz_RUNTIME_DIR}" ENV PATH
                NO_CACHE)
        else ()
            find_library(_graphviz_file NAMES "${_name}"
                HINTS "${Graphviz_RUNTIME_DIR}" ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
                NO_CACHE)
        endif ()

        if (NOT _graphviz_file)
            set(Graphviz_Runtime_FOUND FALSE)
            list(APPEND _graphviz_missing "${_name}")
        endif ()
    endforeach ()

    list(JOIN _graphviz_missing ", " _graphviz_reason)
    find_package_handle_standard_args(Graphviz HANDLE_COMPONENTS
        REASON_FAILURE_MESSAGE "Missing Graphviz runtime libraries: ${_graphviz_reason}")
    return()
endif ()

if (NOT TARGET Graphviz::Graphviz)
    if (WIN32)
        # Without pkg-config on Windows.
        find_path(GRAPHVIZ_INCLUDE_DIR graphviz/gvc.h
            HINTS "$ENV{ProgramFiles}/Graphviz/include")
        find_library(GRAPHVIZ_GVC_LIBRARY gvc
            HINTS "$ENV{ProgramFiles}/Graphviz/lib")
        find_library(GRAPHVIZ_CGRAPH_LIBRARY cgraph
            HINTS "$ENV{ProgramFiles}/Graphviz/lib")

        find_package_handle_standard_args(Graphviz
            REQUIRED_VARS GRAPHVIZ_INCLUDE_DIR GRAPHVIZ_GVC_LIBRARY GRAPHVIZ_CGRAPH_LIBRARY)

        if (Graphviz_FOUND)
            get_filename_component(GRAPHVIZ_ROOT_DIR "${GRAPHVIZ_INCLUDE_DIR}" DIRECTORY)
            set(GRAPHVIZ_BIN_DIR "${GRAPHVIZ_ROOT_DIR}/bin" CACHE PATH "Directory with Graphviz DLLs and plugins")
            add_library(Graphviz::Graphviz INTERFACE IMPORTED)
            set_target_properties(Graphviz::Graphviz PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${GRAPHVIZ_INCLUDE_DIR};${GRAPHVIZ_INCLUDE_DIR}/graphviz"
                INTERFACE_LINK_LIBRARIES "${GRAPHVIZ_GVC_LIBRARY};${GRAPHVIZ_CGRAPH_LIBRARY}")
        endif ()
    else ()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRAPHVIZ REQUIRED IMPORTED_TARGET libgvc libcgraph)
        add_library(Graphviz::Graphviz INTERFACE IMPORTED)
        target_link_libraries(Graphviz::Graphviz INTERFACE PkgConfig::GRAPHVIZ)
        set(Graphviz_FOUND TRUE)
    endif ()
endif ()
