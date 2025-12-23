/**
 * @file synapse_traits.cpp
 * @brief Python bindings for synapse traits.
 * @kaspersky_support Artiom N.
 * @date 21.02.24
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

#include "common.h"
#include "exports.h"


BOOST_PYTHON_MODULE(KNP_FULL_LIBRARY_NAME)
{
    export_delta();
    export_resource_stdp_delta();
}
