/**
 * @file cell_traits.h
 * @brief Common cell registry and parameter selection for population containers.
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

#include <knp/astrocyte-traits/all_traits.h>
#include <knp/neuron-traits/all_traits.h>

namespace knp::core
{

/**
 * @brief True if the cell type is a neuron.
 *
 * @tparam CellType cell type.
 */
template <typename CellType>
constexpr bool is_neuron_v = boost::mp11::mp_contains<neuron_traits::AllNeurons, CellType>::value;

/**
 * @brief True if the cell type is an astrocyte.
 *
 * @tparam CellType cell type.
 */
template <typename CellType>
constexpr bool is_astrocyte_v = boost::mp11::mp_contains<astrocyte_traits::AllAstrocytes, CellType>::value;

/**
 * @brief List of neuron and astrocyte types.
 *
 * @note Neurons precede astrocytes. The order determines population variant indices and SONATA type IDs.
 */
using AllCells = boost::mp11::mp_append<neuron_traits::AllNeurons, astrocyte_traits::AllAstrocytes>;

/**
 * @brief Parameter and default-value types for a cell model.
 *
 * @tparam CellType neuron or astrocyte type.
 * @tparam IsNeuron true for neuron types.
 */
template <typename CellType, bool IsNeuron = is_neuron_v<CellType>>
struct CellTraits;

/**
 * @brief Parameter types for a registered neuron model.
 */
template <typename CellType>
struct CellTraits<CellType, true>
{
    /**
     * @brief Cell parameters.
     */
    using Parameters = neuron_traits::neuron_parameters<CellType>;
    /**
     * @brief Default parameter values.
     */
    using DefaultValues = neuron_traits::default_values<CellType>;
};

/**
 * @brief Parameter types for a registered astrocyte model.
 */
template <typename CellType>
struct CellTraits<CellType, false>
{
    static_assert(is_astrocyte_v<CellType>, "Cell type must be registered as a neuron or an astrocyte.");
    /**
     * @brief Cell parameters.
     */
    using Parameters = astrocyte_traits::astrocyte_parameters<CellType>;
    /**
     * @brief Default parameter values.
     */
    using DefaultValues = astrocyte_traits::default_values<CellType>;
};

}  // namespace knp::core
