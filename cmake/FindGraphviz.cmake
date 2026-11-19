# - Find Graphviz.

include(FindPackageHandleStandardArgs)

if (NOT TARGET Graphviz::Graphviz)
    if (WIN32)
        # Without pkg-config on Windows.
        find_path(Graphviz_INCLUDE_DIR graphviz/gvc.h
            HINTS "$ENV{ProgramFiles}/Graphviz/include")
        find_library(Graphviz_GVC_LIBRARY gvc
            HINTS "$ENV{ProgramFiles}/Graphviz/lib")
        find_library(Graphviz_CGRAPH_LIBRARY cgraph
            HINTS "$ENV{ProgramFiles}/Graphviz/lib")

        find_package_handle_standard_args(Graphviz
            REQUIRED_VARS Graphviz_INCLUDE_DIR Graphviz_GVC_LIBRARY Graphviz_CGRAPH_LIBRARY)

        if (Graphviz_FOUND)
            add_library(Graphviz::Graphviz INTERFACE IMPORTED)
            set_target_properties(Graphviz::Graphviz PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${Graphviz_INCLUDE_DIR};${Graphviz_INCLUDE_DIR}/graphviz"
                INTERFACE_LINK_LIBRARIES "${Graphviz_GVC_LIBRARY};${Graphviz_CGRAPH_LIBRARY}")
        endif ()
    else ()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRAPHVIZ REQUIRED IMPORTED_TARGET libgvc libcgraph)
        add_library(Graphviz::Graphviz INTERFACE IMPORTED)
        target_link_libraries(Graphviz::Graphviz INTERFACE PkgConfig::GRAPHVIZ)
        set(Graphviz_FOUND TRUE)
    endif ()
endif ()