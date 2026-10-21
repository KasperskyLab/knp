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

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "network_graph_impl.h"


namespace knp::framework::visualizer::impl
{

/**
 * @brief Add node to the graph and index it by UID.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraphImpl::vertex_descriptor NetworkGraphImpl::add_node(const Node& node)
{
    auto v = boost::add_vertex(node, graph_);
    if (!node.is_invisible()) map_uid_to_nodes_descriptor[node.get_uid()] = v;
    return v;
}


/**
 * @brief Add edge to the graph and index it by UID.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraphImpl::edge_descriptor NetworkGraphImpl::add_edge(
    vertex_descriptor from, vertex_descriptor to, const Edge& edge)
{
    auto e = boost::add_edge(from, to, edge, graph_).first;
    map_uid_to_edges_descriptor.emplace(edge.get_uid(), e);
    return e;
}


/**
 * @brief Add invisible node for an edge without source or destination.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraphImpl::vertex_descriptor NetworkGraphImpl::add_placeholder_node()
{
    // A default UID is a new random one, as for a default node.
    Node ghost_node(
        knp::core::UID{}, "ghost_node_" + std::to_string(++ghost_counter_), std::nullopt, std::nullopt, "ghost");
    ghost_node.set_invisible(true);
    return add_node(ghost_node);
}


/**
 * @brief Find node by UID.
 *
 * @details Parameters and return value are described in the header.
 */
std::optional<NetworkGraphImpl::vertex_descriptor> NetworkGraphImpl::find_node(const knp::core::UID& uid) const
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
std::vector<NetworkGraphImpl::edge_descriptor> NetworkGraphImpl::find_edges(const knp::core::UID& uid) const
{
    std::vector<edge_descriptor> result;
    auto [first, last] = map_uid_to_edges_descriptor.equal_range(uid);
    for (auto it = first; it != last; ++it) result.push_back(it->second);
    return result;
}

}  // namespace knp::framework::visualizer::impl


namespace knp::framework::visualizer
{

/**
 * @brief Network graph constructor with an implementation.
 *
 * @details Parameters are described in the header.
 */
NetworkGraph::NetworkGraph(std::unique_ptr<impl::NetworkGraphImpl>&& impl) : impl_(std::move(impl))
{
    if (!impl_) throw std::invalid_argument("Network graph implementation is not set.");
}


/**
 * @brief Move constructor.
 */
NetworkGraph::NetworkGraph(NetworkGraph&&) noexcept = default;


/**
 * @brief Move assignment operator.
 *
 * @details Return value is described in the header.
 */
NetworkGraph& NetworkGraph::operator=(NetworkGraph&&) noexcept = default;


/**
 * @brief Destructor. It is defined here, where the implementation class is complete.
 */
NetworkGraph::~NetworkGraph() = default;


/**
 * @brief Check that the graph was built with message bus subscriptions.
 *
 * @details Parameters and return value are described in the header.
 */
bool NetworkGraph::is_dynamic() const
{
    return impl_->is_dynamic();
}


/**
 * @brief Get network graph implementation.
 *
 * @details Parameters and return value are described in the header.
 */
const impl::NetworkGraphImpl& NetworkGraph::get_impl() const
{
    return *impl_;
}

}  // namespace knp::framework::visualizer
