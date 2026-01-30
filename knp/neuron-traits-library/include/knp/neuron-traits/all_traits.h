/**
 * @file all_traits.h
 * @brief List of all neuron type traits.
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

#include <knp/meta/stringify.h>

#include <boost/mp11.hpp>

#include "altai_lif.h"
#include "blifat.h"
#include "stdp_synaptic_resource_rule.h"
#include "stdp_type_traits.h"

/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{
/**
 * @brief Comma-separated list of neuron tags.
 */
#define ALL_NEURONS BLIFATNeuron, SynapticResourceSTDPBLIFATNeuron, AltAILIF, SynapticResourceSTDPAltAILIFNeuron


/**
 * @brief List of neuron types.
 * @details To add a new neuron type to the list, define it in the ALL_NEURONS macro. For example,
 * @code{.cpp}
 * #define ALL_NEURONS BLIFATNeuron, IzhikevichNeuron
 * @endcode
 */
using AllNeurons = boost::mp11::mp_list<ALL_NEURONS>;


/**
 * @brief A tuple that contains string names of neuron types.
 */
constexpr auto neurons_names = KNP_MAKE_TUPLE(ALL_NEURONS);

}  // namespace knp::neuron_traits
