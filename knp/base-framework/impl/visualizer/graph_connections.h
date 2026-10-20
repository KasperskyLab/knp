/**
 * @file graph_connections.h
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

#pragma once

#include <knp/core/uid.h>
#include <knp/framework/network.h>
#include <knp/framework/visualizer/dynamic_info.h>
#include <knp/framework/visualizer/network_graph.h>

#include <spdlog/spdlog.h>

#include <any>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>


/**
 * @brief Implementation namespace of the network visualizer.
 */
namespace knp::framework::visualizer::impl
{

/**
 * @brief Connections of edges: edge position in the edge vector to a set of node UIDs.
 *
 * @details The key is a position and not a UID, because an edge can have no meaningful UID of its own, and several
 * entities can have the same UID. A set is needed because one projection can have several senders or several
 * receivers on the bus.
 */
using EdgeConnections = std::map<size_t, std::set<knp::core::UID>>;


/**
 * @brief Positions of entities in a node or an edge vector, by UID.
 */
using UidIndex = std::map<knp::core::UID, size_t>;


/**
 * @brief Length of a short name made from UID.
 *
 * @details The first group of a UUID in its text form has 8 hexadecimal digits. It is enough to tell entities apart on
 * a picture and keeps labels short.
 */
constexpr size_t uid_part_size = 8;


/**
 * @brief Get entity name for any network object. If there is no name it's constructed from UID.
 *
 * @param pop population or projection object.
 *
 * @return string name of the entity.
 */
template <class Entity>
std::string get_population_name(const Entity& pop)
{
    // A copy: TagMap::get_tag() is not a constant method and inserts an empty value for an unknown key.
    knp::core::TagMap tags = std::visit([](const auto& p) { return p.get_tags(); }, pop);
    knp::core::UID uid = std::visit([](const auto& p) { return p.get_uid(); }, pop);
    std::string name;
    auto tag = tags.get_tag("name");
    if (tag.has_value())
    {
        try
        {
            name = std::any_cast<std::string>(tag);
        }
        catch (const std::bad_any_cast&)
        {
            SPDLOG_ERROR("Wrong name tag type for entity {}.", std::string(uid));
        }
    }
    if (name.empty()) name = std::string{uid}.substr(0, uid_part_size);
    return name;
}


/**
 * @brief Get nodes and edges of a network.
 *
 * @param network source network.
 * @param nodes vector of nodes to fill.
 * @param edges vector of edges to fill.
 * @param node_src map of edge sources to fill.
 * @param node_dst map of edge destinations to fill.
 */
void get_static_connections(
    const knp::framework::Network& network, std::vector<Node>& nodes, std::vector<Edge>& edges,
    EdgeConnections& node_src, EdgeConnections& node_dst);


/**
 * @brief Add nodes and edges taken from message endpoint subscriptions and from the calling code.
 *
 * @param info bus connections and entities that cannot be restored from the bus.
 * @param nodes vector of nodes to extend.
 * @param edges vector of edges to extend.
 * @param node_src map of edge sources to extend.
 * @param node_dst map of edge destinations to extend.
 *
 * @details The function is called after get_static_connections, so an entity that is already known from the network
 * is not added twice.
 */
void get_dynamic_connections(
    const DynamicInfo& info, std::vector<Node>& nodes, std::vector<Edge>& edges, EdgeConnections& node_src,
    EdgeConnections& node_dst);

}  // namespace knp::framework::visualizer::impl
