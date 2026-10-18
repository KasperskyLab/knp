/**
 * @file network_graph.cpp
 * @brief Network graph storage with UID indexes.
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

#include <knp/framework/visualizer/network_graph.h>

#include <optional>
#include <string>
#include <vector>


namespace knp::framework::visualizer
{

/**
 * @brief Add node to the graph and index it by UID.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph::vertex_descriptor NetworkGraph::add_node(const Node& node)
{
    auto v = boost::add_vertex(node, graph_);
    if (!node.is_invisible_) map_uid_to_nodes_descriptor[node.uid_] = v;
    return v;
}


/**
 * @brief Add edge to the graph and index it by UID.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph::edge_descriptor NetworkGraph::add_edge(vertex_descriptor from, vertex_descriptor to, const Edge& edge)
{
    auto e = boost::add_edge(from, to, edge, graph_).first;
    map_uid_to_edges_descriptor.emplace(edge.uid_, e);
    return e;
}


/**
 * @brief Add invisible node for an edge without source or destination.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph::vertex_descriptor NetworkGraph::add_placeholder_node()
{
    Node ghost_node;
    ghost_node.name_ = "ghost_node_" + std::to_string(++ghost_counter_);
    ghost_node.comment_ = "ghost";
    ghost_node.is_invisible_ = true;
    return add_node(ghost_node);
}


/**
 * @brief Find node by UID.
 *
 * @details Parameters and return value are described in the header.
 */
std::optional<NetworkGraph::vertex_descriptor> NetworkGraph::find_node(const knp::core::UID& uid) const
{
    auto it = map_uid_to_nodes_descriptor.find(uid);
    if (it == map_uid_to_nodes_descriptor.end()) return std::nullopt;
    return it->second;
}


/**
 * @brief Find all edges of a projection.
 *
 * @details Parameters and return value are described in the header.
 */
std::vector<NetworkGraph::edge_descriptor> NetworkGraph::find_edges(const knp::core::UID& uid) const
{
    std::vector<edge_descriptor> result;
    auto [first, last] = map_uid_to_edges_descriptor.equal_range(uid);
    for (auto it = first; it != last; ++it) result.push_back(it->second);
    return result;
}

}  // namespace knp::framework::visualizer
