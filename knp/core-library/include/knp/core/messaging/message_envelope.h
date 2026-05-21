/**
 * @file message_envelope.h
 * @brief Message envelope routines.
 * @kaspersky_support Artiom N.
 * @date 13.04.2023
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

#include <knp/core/uid.h>

#include <iostream>
#include <variant>
#include <vector>

#include "messaging.h"


/**
 * @brief Messaging namespace.
 */
namespace knp::core::messaging
{

/**
 * @brief Message variant that contains any message type specified in `AllMessages`.
 * 
 * @details `MessageVariant` takes the value of `std::variant<MessageType_1,..., MessageType_n>`, where
 * `MessageType_[1..n]` is the message type specified in `AllMessages`. \n For example, if `AllMessages` contains
 * SpikeMessage and SynapticImpactMessage types, then `MessageVariant = std::variant<SpikeMessage,
 * SynapticImpactMessage>`. \n `MessageVariant` retains the same order of message types as defined in `AllMessages`.
 * 
 * @see ALL_MESSAGES.
 */
using MessageVariant = boost::mp11::mp_rename<AllMessages, std::variant>;


/**
 * @brief Pack messages to envelope.
 * 
 * @param message message to pack.
 * 
 * @return vector with data of a serialized message.
 */
std::vector<uint8_t> pack_to_envelope(const MessageVariant &message);
/**
 * @brief Extract messages from envelope.
 * 
 * @param buffer message buffer.
 * 
 * @return `std::variant` of message types.
 */
MessageVariant extract_from_envelope(const void *buffer);
/**
 * @brief Extract messages from envelope.
 * 
 * @param buffer message buffer in the `uint8_t` vector format.
 * 
 * @return `std::variant` of message types.
 */
MessageVariant extract_from_envelope(const std::vector<uint8_t> &buffer);

}  // namespace knp::core::messaging
