/**
 * @file graph_helpers.cpp
 * @brief Help functions for graph vizualize.
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

#include "graph_helpers.h"

#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>


/**
 * @brief Get bus messages from backend subscriptions.
 *
 * @param backend shared pointer to backend.
 *
 * @return tuple of spike messages and synaptic messages.
 */
using SpikeMessages = std::vector<std::pair<knp::core::UID, knp::core::UID>>;
using SynapticMessages = std::vector<std::pair<knp::core::UID, knp::core::UID>>;
using BusMessages = std::tuple<SpikeMessages, SynapticMessages>;


BusMessages get_bus(std::shared_ptr<knp::core::Backend>& backend)
{
    const auto subs = backend->get_message_endpoint().get_endpoint_subscriptions();

    constexpr size_t spike_idx = 0;            // Must be 0 (Serial number of the type).
    constexpr size_t synaptic_impact_idx = 1;  // Must be 1 (Serial number of the type).

    SpikeMessages spike_messages;        // [sender, receiver]
    SynapticMessages synaptic_messages;  // [sender, receiver]


    // take from bus nessosary maps
    for (const auto& [key, sub_variant] : subs)
    {
        // const auto& [type_idx, receiver_uid] = key;
        const auto& type_idx = key.first;
        const auto& receiver_uid = key.second;
        std::visit(
            [&](const auto& sub)
            {
                const auto& senders = sub.get_senders();
                if (senders.empty()) return;
                if (type_idx == spike_idx)
                {
                    // Receiver: Projection, sender: *Population (* - or modificator) (0 -->).
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(spike_messages),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
                else if (type_idx == synaptic_impact_idx)
                {
                    // receiver: Population,      sender: Projection (* - or modificator) (--> 0).
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(synaptic_messages),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
            },
            sub_variant);
    }

    return {spike_messages, synaptic_messages};
}


/**
 * @brief Get node name by UID from network graph nodes.
 *
 * @param node_uid UID of the node.
 * @param nodes vector of network graph nodes.
 *
 * @return string node name.
 */
std::string get_node_name(knp::core::UID node_uid, std::vector<knp::framework::NetworkGraph::Node> const& nodes)
{
    std::string node_name = std::string(node_uid).substr(0, 8);
    // nodes from netGraph
    auto node_it =
        std::find_if(nodes.begin(), nodes.end(), [&node_uid](const auto& node) { return node.uid_ == node_uid; });
    if (node_it != nodes.end()) node_name = node_it->name_;
    return node_name;
}


/**
 * @brief Get graph node by UID from network graph nodes.
 *
 * @param node_uid UID of the node.
 * @param nodes vector of network graph nodes.
 *
 * @return NetworkGraph::Node object.
 */
knp::framework::NetworkGraph::Node get_graph_node_by_uid(
    knp::core::UID node_uid, std::vector<knp::framework::NetworkGraph::Node> const& nodes)
{
    auto node_it =
        std::find_if(nodes.begin(), nodes.end(), [&node_uid](const auto& node) { return node.uid_ == node_uid; });
    if (node_it != nodes.end())
    {
        return *node_it;
    }

    SPDLOG_INFO("Node with UID {} not found in graph", std::string(node_uid));
    return knp::framework::NetworkGraph::Node{};
}


/**
 * @brief Get projection size by UID from network graph edges.
 *
 * @param edges vector of network graph edges.
 * @param edge_uid UID of the edge.
 *
 * @return string size of the edge.
 */
std::string get_proj_size(std::vector<knp::framework::NetworkGraph::Edge> const& edges, const knp::core::UID& edge_uid)
{
    auto it = std::find_if(
        edges.begin(), edges.end(), [&edge_uid](const auto& graph_edge) { return graph_edge.uid_ == edge_uid; });
    if (it != edges.end())
    {
        return std::to_string(it->size_);
    }
    throw std::runtime_error("Edge not found");
}
