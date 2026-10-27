/**
 * @file dynamic_info.h
 * @brief Data for a dynamic network graph: message bus connections and external entities.
 * @kaspersky_support Kirill L.
 * @date 05.08.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <string>
#include <utility>
#include <vector>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Connections taken from message endpoint subscriptions.
 *
 * @details Plain copies of the subscriptions: [sender, receiver] pairs. The structure keeps no reference to a
 * backend, so it stays valid after the backend is destroyed and can be passed anywhere.
 */
struct BusConnections
{
    /// Spike messages: the sender is a population, a handler or an input channel, the receiver is a projection.
    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<knp::core::UID, knp::core::UID>> spike_messages_;
    /// Synaptic impact messages: the sender is a projection, the receiver is a population.
    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<knp::core::UID, knp::core::UID>> synaptic_messages_;
};


/**
 * @brief Entity that is not described by the network: a message handler or a channel.
 *
 * @details Some entities cannot be restored from the message bus of a backend. A message handler subscribes to its
 * senders through its own message endpoint, and an output channel does the same, so these subscriptions are not
 * registered in the backend. Such entities are described by the calling code.
 */
struct ExternalEntity
{
    /// Entity UID.
    knp::core::UID uid_;
    /// Node kind for the label, for example "WTA", "Input channel" or "Output channel".
    std::string name_ = "External";
    /// Entities the given one listens to. Leave empty if the subscriptions are visible on the bus.
    // cppcheck-suppress unusedStructMember
    std::vector<knp::core::UID> senders_;
};


/**
 * @brief Data for a dynamic graph.
 */
struct DynamicInfo
{
    /// Connections taken from the backend message bus.
    // cppcheck-suppress unusedStructMember
    BusConnections bus_;
    /// Handlers and channels. An entity that is not listed here is drawn as "External".
    // cppcheck-suppress unusedStructMember
    std::vector<ExternalEntity> entities_;
};

}  // namespace knp::framework::visualizer
