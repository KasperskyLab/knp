/**
 * @file type_traits.h
 * @brief Common declarations for astrocyte model parameters.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <map>
#include <string>
#include <vector>

/**
 * @brief Namespace for astrocyte traits.
 */
namespace knp::astrocyte_traits
{

/**
 * @brief Structure for astrocyte parameters.
 *
 * @tparam AstrocyteType astrocyte type.
 */
template <typename AstrocyteType>
struct astrocyte_parameters;


/**
 * @brief Structure for astrocyte default values.
 *
 * @tparam AstrocyteType astrocyte type.
 */
template <typename AstrocyteType>
struct default_values;

/**
 * @brief Population tag containing UIDs of delta-synapse projections controlled by an astrocyte population.
 *
 * @details The tag stores `std::vector<knp::core::UID>`. Without the tag, the astrocyte tracks calcium
 * without modulating projections.
 */
inline constexpr char astrocyte_modulated_projections_tag[] = "astrocyte_modulated_projections";

/**
 * @brief Original synapse weights indexed by controlled projection UID strings.
 */
using AstrocyteProjectionWeights = std::map<std::string, std::vector<float>>;

/**
 * @brief Population tag storing AstrocyteProjectionWeights for multiplicative modulation.
 */
inline constexpr char astrocyte_original_weights_tag[] = "astrocyte_original_weights";

/**
 * @brief Projection tag storing original weights shared by its controlling astrocyte populations.
 */
inline constexpr char astrocyte_projection_weights_tag[] = "astrocyte_projection_weights";


}  // namespace knp::astrocyte_traits
