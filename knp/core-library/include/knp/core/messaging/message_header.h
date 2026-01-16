/**
 * @file message_header.h
 * @brief Message header class.
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

#include <knp/core/core.h>
#include <knp/core/uid.h>

#include <iostream>
#include <vector>


/**
 * @brief Messaging namespace.
 */
namespace knp::core::messaging
{
/**
 * @brief Common header for messages.
 */
struct MessageHeader
{
    /**
     * @brief UID of the object that sent the message.
     */
    knp::core::UID sender_uid_;
    /**
     * @brief Index of the network execution step.
     */
    Step send_time_;
};


/**
 * @brief Get message header from an input stream.
 * @param stream input stream.
 * @param header message header to get from the input stream.
 * @return input stream.
 */
std::istream &operator>>(std::istream &stream, MessageHeader &header);

/**
 * @brief Send message header to an output stream.
 * @param stream output stream.
 * @param header message header to send to the output stream.
 * @return output stream.
 */
std::ostream &operator<<(std::ostream &stream, const MessageHeader &header);

}  // namespace knp::core::messaging
