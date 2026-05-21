/**
 * @file type_traits.h
 * @brief Common synapse type traits.
 * @kaspersky_support Artiom N.
 * @date 26.01.2023
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
 * @brief Structure for synapse parameters.
 * 
 * @tparam SynapseType type of synapses.
 */
template <typename SynapseType>
struct synapse_parameters;


/**
 * @brief Structure for synapse default values.
 * 
 * @tparam SynapseType type of synapses.
 */
template <typename SynapseType>
struct default_values;


/**
 * @brief Structure for parameters shared between synapses.
 * 
 * @tparam SynapseType type of synapses.
 */
template <typename SynapseType>
struct shared_synapse_parameters
{
};
}  // namespace knp::synapse_traits
