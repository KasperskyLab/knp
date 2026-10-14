/**
 * @file graph_helpers.cpp
 * @brief Collecting nodes and edges of a network and of a message bus.
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

#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>


namespace
{

constexpr size_t uid_part_size = 8;

/// Node kind of an entity that is not described by the network and is not described by the calling code.
constexpr const char* external_node_type = "External";

/// Edge kind of a spike subscription that is not a projection.
constexpr const char* message_edge_type = "Spike Message";

/// Entities described by the calling code, by UID.
using KnownEntities = std::map<knp::core::UID, const knp::framework::ExternalEntity*>;

/**
 * @brief Check that UID is nil.
 *
 * @param uid UID to check.
 *
 * @return boolean indicating that the UID is not set.
 */
bool is_nil_uid(const knp::core::UID& uid)
{
    // UID::operator bool() returns false for a nil UUID.
    return !static_cast<bool>(uid);
}


/**
 * @brief Make a short readable name from UID.
 *
 * @param uid UID of an entity.
 *
 * @return first characters of the UID.
 */
std::string get_short_name(const knp::core::UID& uid)
{
    return std::string{uid}.substr(0, uid_part_size);
}


/**
 * @brief Make an index of entity UIDs to their positions in a vector.
 *
 * @param items vector of nodes or edges.
 *
 * @return map of UID to position. An entity with a nil UID is not indexed.
 */
template <class Item>
UidIndex make_index(const std::vector<Item>& items)
{
    UidIndex index;
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (is_nil_uid(items[i].uid_)) continue;
        index.emplace(items[i].uid_, i);
    }
    return index;
}


/**
 * @brief Make an edge for a projection that is not described by the network.
 *
 * @param uid UID of the projection.
 *
 * @return edge object.
 */
knp::framework::Edge make_bus_edge(const knp::core::UID& uid)
{
    knp::framework::Edge edge{uid,          get_short_name(uid), std::nullopt,    std::nullopt,
                              std::nullopt, std::nullopt,        "not in network"};
    edge.is_dynamic_ = true;
    return edge;
}


/**
 * @brief Make an edge for a spike subscription that is not a projection.
 *
 * @return edge object.
 */
knp::framework::Edge make_message_edge()
{
    knp::framework::Edge edge;
    // The name is empty: a spike subscription has nothing to name, the kind is written alone.
    edge.edge_type_ = message_edge_type;
    edge.is_dynamic_ = true;
    return edge;
}

}  // namespace


namespace knp::framework
{

/**
 * @brief Copy subscriptions of a message endpoint.
 *
 * @details Parameters and return value are described in the header.
 */
BusConnections get_bus_connections(knp::core::MessageEndpoint& endpoint)
{
    const auto subs = endpoint.get_endpoint_subscriptions();

    constexpr size_t spike_idx = 0;            // Must be 0 (Serial number of the type).
    constexpr size_t synaptic_impact_idx = 1;  // Must be 1 (Serial number of the type).

    knp::framework::BusConnections connections;

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
                if (type_idx == spike_idx)
                {
                    // Receiver: projection or spike consumer, sender: population, handler or channel (0 -->).
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(connections.spike_messages_),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
                else if (type_idx == synaptic_impact_idx)
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

}  // namespace knp::framework


/**
 * @brief Get nodes and edges of a network.
 *
 * @details Parameters and return value are described in the header.
 */
void get_static_connections(
    const knp::framework::Network& network, std::vector<knp::framework::Node>& nodes,
    std::vector<knp::framework::Edge>& edges, EdgeConnections& node_src, EdgeConnections& node_dst)
{
    // Add populations as nodes.
    for (const auto& pop : network.get_populations())
    {
        size_t pop_size = std::visit([](const auto& p) { return p.size(); }, pop);
        knp::core::UID uid = std::visit([](const auto& p) { return p.get_uid(); }, pop);
        nodes.push_back(knp::framework::Node{uid, get_population_name(pop), pop_size, pop.index(), std::nullopt});
    }

    // Add projections as edges.
    for (const auto& proj : network.get_projections())
    {
        size_t proj_size = std::visit([](const auto& p) { return p.size(); }, proj);
        knp::core::UID uid = std::visit([](const auto& p) { return p.get_uid(); }, proj);
        knp::core::UID uid_from = std::visit([](const auto& p) { return p.get_presynaptic(); }, proj);
        knp::core::UID uid_to = std::visit([](const auto& p) { return p.get_postsynaptic(); }, proj);

        knp::framework::Edge edge{
            uid, get_population_name(proj), std::nullopt, std::nullopt, proj_size, proj.index(), std::nullopt};
        edge.is_static_ = true;
        edges.push_back(edge);

        const size_t edge_position = edges.size() - 1;
        if (!is_nil_uid(uid_from)) node_src[edge_position].insert(uid_from);
        if (!is_nil_uid(uid_to)) node_dst[edge_position].insert(uid_to);
    }
}


/**
 * @brief Add nodes and edges taken from message endpoint subscriptions and from the calling code.
 *
 * @details Parameters and return value are described in the header.
 */
void get_dynamic_connections(
    const knp::framework::DynamicInfo& info, std::vector<knp::framework::Node>& nodes,
    std::vector<knp::framework::Edge>& edges, EdgeConnections& node_src, EdgeConnections& node_dst)
{
    // Entities described by the calling code: handlers and channels.
    KnownEntities known_entities;
    for (const auto& entity : info.entities_) known_entities.emplace(entity.uid_, &entity);

    // Positions of entities that are already known from the network.
    UidIndex node_index = make_index(nodes);
    UidIndex edge_index = make_index(edges);

    // Add a node for an entity that is not described by the network. Its kind is taken from the calling code, so
    // nothing is guessed here: an entity that is not described there gets the "External" kind.
    auto add_bus_node = [&](const knp::core::UID& uid)
    {
        auto it = node_index.find(uid);
        if (it != node_index.end()) return it->second;

        knp::framework::Node node{uid, get_short_name(uid), std::nullopt, std::nullopt, "not in network"};
        node.is_static_ = false;
        auto entity_it = known_entities.find(uid);
        node.node_type_ = entity_it != known_entities.end() ? entity_it->second->name_ : external_node_type;

        nodes.push_back(node);
        node_index[uid] = nodes.size() - 1;
        return nodes.size() - 1;
    };

    // Synaptic impact message: sender is a projection, receiver is a population (--> 0).
    // This loop goes first to register projections that are not described by the network.
    for (const auto& [sender, receiver] : info.bus_.synaptic_messages_)
    {
        auto it = edge_index.find(sender);
        if (it == edge_index.end())
        {
            SPDLOG_INFO("Projection {} is known from the bus only.", std::string(sender));
            edges.push_back(make_bus_edge(sender));
            it = edge_index.emplace(sender, edges.size() - 1).first;
        }
        else
        {
            edges[it->second].is_dynamic_ = true;
        }

        node_dst[it->second].insert(receiver);
        add_bus_node(receiver);
    }

    // Spike message: sender is a population, a handler or a channel, receiver is a projection or a spike consumer.
    for (const auto& [sender, receiver] : info.bus_.spike_messages_)
    {
        add_bus_node(sender);

        auto it = edge_index.find(receiver);
        if (it != edge_index.end())
        {
            node_src[it->second].insert(sender);
            edges[it->second].is_dynamic_ = true;
            continue;
        }

        // The receiver is not a projection: it is an output channel or an observer subscribed to spikes directly.
        add_bus_node(receiver);
        edges.push_back(make_message_edge());
        const size_t edge_position = edges.size() - 1;
        node_src[edge_position].insert(sender);
        node_dst[edge_position].insert(receiver);
    }

    // A handler and an output channel subscribe to their senders through their own message endpoints. Such a
    // subscription is not registered in the backend and dies with the entity that made it, so these connections
    // exist only where they were created and are taken from the calling code.
    for (const auto& entity : info.entities_)
    {
        if (entity.senders_.empty()) continue;

        add_bus_node(entity.uid_);
        for (const auto& sender : entity.senders_)
        {
            add_bus_node(sender);
            edges.push_back(make_message_edge());
            const size_t edge_position = edges.size() - 1;
            node_src[edge_position].insert(sender);
            node_dst[edge_position].insert(entity.uid_);
        }
    }
}
