/**
 * @file stdp_type_traits.h
 * @brief List of STDP-specific neuron type traits.
 * @kaspersky_support A. Vartenkov.
 * @date 7.12.2023
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
#include "altai_lif.h"
#include "blifat.h"
#include "stdp_synaptic_resource_rule.h"

/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{
/**
 * @brief BLIFAT neuron with additional resource-based STDP parameters.
 */
using SynapticResourceSTDPBLIFATNeuron = SynapticResourceSTDPNeuron<BLIFATNeuron>;

/**
 * @brief AltAILIF neuron with additional resource-based STDP parameters.
 */
using SynapticResourceSTDPAltAILIFNeuron = SynapticResourceSTDPNeuron<AltAILIF>;

}  // namespace knp::neuron_traits
