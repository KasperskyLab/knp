/**
 * @file spike_message.h
 * @brief Spike message class.
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

#include <iostream>
#include <vector>

#include "message_header.h"


/**
 * @brief Messaging namespace.
 */
namespace knp::core::messaging
{

// TODO: bind this to population size. Don't forget to change protobuf if changing this.
/**
 * @brief Spike index type in the form of a 32-bit unsigned integer.
 */
using SpikeIndex = uint32_t;


/**
 * @brief List of spike indexes.
 */
using SpikeData = std::vector<SpikeIndex>;


/**
 * @brief Structure of the spike message.
 */
struct SpikeMessage
{
    /**
     * @brief Message header.
     */
    MessageHeader header_;

    /**
     * @brief Indexes of the recently spiked neurons.
     */
    SpikeData neuron_indexes_;

    /**
     * @todo Maybe add operator `[]` and others to be able to use templates for message processing.
     */
};


/**
 * @brief Check if two spike messages are the same.
 * 
 * @param sm1 first message.
 * @param sm2 second message.
 * 
 * @return `true` if both messages are the same.
 */
bool operator==(const SpikeMessage &sm1, const SpikeMessage &sm2);


/**
 * @brief Send spiking neuron indexes to an output stream.
 * 
 * @param stream output stream.
 * @param spikes index vector.
 * 
 * @return output stream.
 */
std::ostream &operator<<(std::ostream &stream, const core::messaging::SpikeData &spikes);

/**
 * @brief Send spike message to an output stream.
 * 
 * @param stream output stream.
 * @param msg spike message to send to the output stream.
 * 
 * @return output stream.
 */
std::ostream &operator<<(std::ostream &stream, const SpikeMessage &msg);

/**
 * @brief Get spike message from an input stream.
 * 
 * @param stream input stream.
 * @param msg spike message to get from the input stream.
 * 
 * @return input stream.
 */
std::istream &operator>>(std::istream &stream, SpikeMessage &msg);

}  // namespace knp::core::messaging
