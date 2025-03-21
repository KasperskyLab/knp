/**
 * @file synaptic_impact_message.h
 * @brief Synaptic impact message class.
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
#include <knp/synapse-traits/output_types.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#include "message_header.h"


/**
 * @brief Messaging namespace.
 */
namespace knp::core::messaging
{

/**
 * @brief Structure that contains the synaptic impact value and indexes of presynaptic and postsynaptic neurons.
 * @details Synaptic impact changes parameters of neurons after the synapses state is calculated.
 */
struct SynapticImpact
{
    /**
     * @brief Index of the population synapse.
     */
    uint64_t connection_index_;

    /**
     * @brief Value used to change neuron membrane potential.
     */
    float impact_value_;

    /**
     * @brief Synapse type that might define the value role inside the neuron function.
     */
    knp::synapse_traits::OutputType synapse_type_;

    /**
     * @brief Index of the presynaptic neuron connected to the synapse.
     */
    uint32_t presynaptic_neuron_index_;

    /**
     * @brief Index of the postsynaptic neuron connected to the synapse.
     */
    uint32_t postsynaptic_neuron_index_;

    /**
     * @brief Compare synaptic impact messages.
     * @return `true` if synaptic impacts are equal.
     */
    bool operator==(const SynapticImpact &) const;
};


/**
 * @brief Structure of the synaptic impact message.
 */
struct SynapticImpactMessage
{
    /**
     * @brief Message header.
     */
    MessageHeader header_;

    /**
     * @brief UID of the population that sends spikes to the projection.
     */
    UID presynaptic_population_uid_;

    /**
     * @brief UID of the population that receives impacts from the projection.
     */
    UID postsynaptic_population_uid_;

    /**
     * @brief Boolean value that defines whether the signal is from a projection without plasticity.
     * @details The parameter is used in training. Use `true` if the signal is from a projection without plasticity.
     * @todo Try to remove this when fixing main; this parameter is too specific to be a part of a general message.
     */
    bool is_forcing_ = false;

    /**
     * @brief Impact values.
     */
    std::vector<SynapticImpact> impacts_;
};


/**
 * @brief Check if two synaptic impact messages are the same.
 * @param sm1 first message.
 * @param sm2 second message.
 * @return `true` if both messages are the same.
 */
bool operator==(const SynapticImpactMessage &sm1, const SynapticImpactMessage &sm2);


/**
 * @brief Get synaptic impact from an input stream.
 * @param stream input stream.
 * @param impact synaptic impact to get from the input stream.
 * @return input stream.
 */
std::istream &operator>>(std::istream &stream, SynapticImpact &impact);


/**
 * @brief Send synaptic impact to an output stream.
 * @param stream output stream.
 * @param impact synaptic impact to send to the output stream.
 * @return output stream.
 */
std::ostream &operator<<(std::ostream &stream, const SynapticImpact &impact);


/**
 * @brief Send synaptic impact message to an output stream.
 * @param stream output stream.
 * @param msg synaptic impact message to send to the output stream.
 * @return output stream.
 */
std::ostream &operator<<(std::ostream &stream, const SynapticImpactMessage &msg);


/**
 * @brief Get synaptic impact message from an input stream.
 * @param stream input stream.
 * @param msg synaptic impact message to get from the input stream.
 * @return input stream.
 */
std::istream &operator>>(std::istream &stream, SynapticImpactMessage &msg);

}  // namespace knp::core::messaging
