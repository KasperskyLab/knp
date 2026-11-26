# - Collect Graphviz runtime.
# Record the names of the Graphviz libraries that are actually needed by the assembled knp-base-framework-core.

file(GET_RUNTIME_DEPENDENCIES
    LIBRARIES "${knp_core}"
    DIRECTORIES ${graphviz_bin}
    PRE_INCLUDE_REGEXES "(^|/)(lib)?(gvc|cgraph|cdt|pathplan|xdot)[.-]"
    PRE_EXCLUDE_REGEXES ".*"
    RESOLVED_DEPENDENCIES_VAR graphviz_libraries)

set(runtime_names "")
foreach(library IN LISTS graphviz_libraries)
    get_filename_component(name "${library}" NAME)
    string(APPEND runtime_names "${name}\n")
endforeach()

file(CONFIGURE OUTPUT "${runtime_manifest}" CONTENT "${runtime_names}" @ONLY)
