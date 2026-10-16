/**
 * @file upcasting.h
 * @brief Converting derived populations and projections into base ones.
 * @kaspersky_support A. Vartenkov
 * @date 19.08.2026
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

#include "knp/framework/network.h"

namespace knp::framework
{
/**
 * @brief Upcast all projections whose synapse type derives from @p BaseSynapseType to the base type.
 *
 * @tparam BaseSynapseType base synapse type to which derived projections will be converted.
 *
 * @param network network whose projections are examined and possibly replaced.
 * 
 * @details The function iterates over every projection in @p network visiting the variant that holds 
 * the concrete projection type. If the projection’s synapse parameters are derived from 
 * `synapse_traits::synapse_parameters<BaseSynapseType>`, the projection is replaced by a new one 
 * created with this function.  Projections that do not satisfy the inheritance condition are left 
 * unchanged.
 */
template <typename SynapseType>
void upcast_projections(Network &network);


/**
 * @brief Upcast all populations whose neuron type derives from @p BaseNeuronType to the base type.

 * @tparam BaseNeuronType base neuron type to which derived populations will be converted.

 * @param network network whose populations are examined and possibly replaced.
 *
 * @details The function walks through every population in @p network visiting the variant that holds 
 * the concrete population type.  If the population’s neuron parameters inherit from 
 * `neuron_traits::neuron_parameters<BaseNeuronType>`, a new population is built with this function 
 * and swapped into the network. Populations that do not satisfy the inheritance check remain unchanged.
 */
template <typename NeuronType>
void upcast_populations(Network &network);

} // namespace knp::framework
