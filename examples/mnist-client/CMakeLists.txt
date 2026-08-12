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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS program_options REQUIRED)

if (NOT TARGET KNP::BaseFramework::Core)
    find_package(knp-base-framework REQUIRED)
endif()

project(mnist-client)
add_executable(mnist-client
    main.cpp
    inference.cpp
    process_data.cpp
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRAPHVIZ REQUIRED IMPORTED_TARGET libgvc libcgraph)

target_include_directories(mnist-client PRIVATE ${GRAPHVIZ_INCLUDE_DIRS})
target_link_libraries(mnist-client PRIVATE PkgConfig::GRAPHVIZ  )
target_link_libraries(mnist-client PRIVATE KNP::BaseFramework::Core ${Boost_LIBRARIES} )
