/**
 * @file bus_connections.cpp
 * @brief Copying subscriptions of a message endpoint.
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

#include <knp/framework/visualizer/graph_builder.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <variant>
#include <vector>


namespace knp::framework::visualizer
{

/**
 * @brief Copy subscriptions of a message endpoint.
 *
 * @details Parameters and return value are described in the header.
 */
BusConnections get_bus_connections(knp::core::MessageEndpoint& endpoint)
{
    const auto subs = endpoint.get_endpoint_subscriptions();

    // A subscription key starts with the index of the message type, the same index the endpoint uses.
    constexpr size_t spike_idx = knp::core::MessageEndpoint::get_type_index<
        knp::core::messaging::MessageVariant, knp::core::messaging::SpikeMessage>;
    constexpr size_t synaptic_impact_idx = knp::core::MessageEndpoint::get_type_index<
        knp::core::messaging::MessageVariant, knp::core::messaging::SynapticImpactMessage>;

    BusConnections connections;

    // Take necessary messages from the bus.
    for (const auto& [key, sub_variant] : subs)
    {
        const auto& type_idx = key.first;
        const auto& receiver_uid = key.second;
        std::visit(
            [&](const auto& sub)
            {
                const auto& senders = sub.get_senders();
                if (senders.empty()) return;
                if (spike_idx == type_idx)
                {
                    // Receiver: projection or spike consumer, sender: population, handler or channel (0 -->).
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(connections.spike_messages_),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
                else if (synaptic_impact_idx == type_idx)
                {
                    // Receiver: population, sender: projection (--> 0).
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(connections.synaptic_messages_),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
            },
            sub_variant);
    }

    return connections;
}

}  // namespace knp::framework::visualizer
