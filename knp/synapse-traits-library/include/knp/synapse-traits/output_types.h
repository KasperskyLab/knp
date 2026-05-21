/**
 * @file output_types.h
 * @brief Enumeration of different ways that a synaptic impact may influence a neuron.
 * @kaspersky_support Vartenkov A.
 * @date 4.03.2023
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

/**
 * @brief Namespace for synapse traits.
 */
namespace knp::synapse_traits
{

/**
 * @brief Types of synapses. This parameter is used for projection-population interaction.
 * 
 * @todo Improve descriptions.
 * Maybe split this enum.
 */
enum class OutputType
{
    /**
     * @brief Excitatory synapse type.
     */
    EXCITATORY = 0,

    /**
     * @brief Inhibitory by current synapse type.
     */
    INHIBITORY_CURRENT = 1,

    /**
     * @brief Inhibitory by conductance synapse type.
     */
    INHIBITORY_CONDUCTANCE = 2,

    /**
     * @brief Dopamine synapse type.
     */
    DOPAMINE = 3,

    /**
     * @brief Neuron-blocking synapse type.
     */
    BLOCKING = 4
};

}  // namespace knp::synapse_traits
