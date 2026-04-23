/**
 * @file generator.h
 * @brief Coordinate generator.
 * @kaspersky_support Artiom N.
 * @date 22.03.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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

#pragma once

#include <any>
#include <functional>


/**
 * @brief Namespace for framework coordinates.
 */
namespace knp::framework::coordinates
{
/**
 * @brief Type of coordinate generator.
 * 
 * @param index neuron index.
 */
using CoordinateGenerator = std::function<std::any(size_t index)>;

}  // namespace knp::framework::coordinates
