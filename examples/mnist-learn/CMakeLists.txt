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
project(mnist_learn_example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost COMPONENTS program_options REQUIRED )

# Need to check if the example build inside KNP tree.
if (NOT TARGET KNP::BaseFramework::Core)
    find_package(knp-base-framework REQUIRED)
endif()

add_executable(mnist_learn_example main.cpp construct_network.cpp train.cpp inference.cpp time_string.cpp)
target_link_libraries(mnist_learn_example PRIVATE KNP::BaseFramework::Core ${Boost_LIBRARIES})
