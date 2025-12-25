/**
 * @file backend.cpp
 * @brief Python bindings for common Backend class.
 * @kaspersky_support Artiom N.
 * @date 01.02.2024
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <filesystem>
#include <string>

#include "backend_wrapper.h"
#include "common.h"

// py::class_<std::vector<core::AllPopulationsVariant>>("AllPopulations")
//     .def(py::vector_indexing_suite<std::vector<core::AllPopulationsVariant>>() );

// py::class_<std::vector<core::AllProjectionsVariant>>("AllProjections")
//     .def(py::vector_indexing_suite<std::vector<core::AllProjectionsVariant>>() );
