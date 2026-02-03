#[[
© 2025 AO Kaspersky Lab

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
project(mnist_learn_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Need to check if the example build inside KNP tree.
if (NOT TARGET KNP::BaseFramework::Core)
    find_package(knp-base-framework REQUIRED)
endif()

find_package(Boost ${KNP_BOOST_MIN_VERSION} COMPONENTS program_options REQUIRED)

add_executable(${PROJECT_NAME} main.cpp model_desc.cpp parse_arguments.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE KNP::BaseFramework::Core Boost::program_options)
