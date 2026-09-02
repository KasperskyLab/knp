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
 * @brief Upcast all synapses derived from SynapseType to SynapseType, leaving other synapses intact.
 * @tparam BaseSynapseType base synapse type.
 * @param network the network to be modified.
 */
template <typename SynapseType>
void upcast_projections(Network &network);


/**
 * @brief Upcast all neurons derived from NeuronType to NeuronType, leaving other synapses intact.
 * @tparam BaseNeuronType base neuron type.
 * @param network the network to be modified.
 */
template <typename NeuronType>
void upcast_populations(Network &network);

} // namespace knp::framework
