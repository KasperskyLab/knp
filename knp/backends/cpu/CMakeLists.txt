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

add_subdirectory(cpu-devices)
add_subdirectory(cpu-library)
add_subdirectory(thread_pool)

file(GLOB backends LIST_DIRECTORIES true "[a-zA-Z0-9_]*-backend")

if(NOT backends)
    message(WARNING "No backend was found.")
    return()
endif()

foreach(backend ${backends})
    message(STATUS "Adding backend \"${backend}\"...")
    get_filename_component(BACKEND_NAME "${backend}" NAME)
    add_subdirectory("${backend}")
endforeach()
