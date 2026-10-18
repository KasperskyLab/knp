/**
 * @file graph_builder.cpp
 * @brief Building a network graph: a Boost Graph from collected nodes and edges.
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

#include "graph_builder.h"

#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <string>
#include <vector>

namespace knp::framework::visualizer::impl
{

/**
 * @brief Get vertex descriptors of entities connected to an edge.
 *
 * @param graph network graph with already added nodes.
 * @param connections map of edge sources or edge destinations.
 * @param edge_position position of the edge in the edge vector.
 *
 * @return vector of vertex descriptors, empty if the edge has no connections on that side.
 */
std::vector<NetworkGraph::vertex_descriptor> get_connected_vertices(
    const NetworkGraph& graph, const EdgeConnections& connections, size_t edge_position)
{
    std::vector<NetworkGraph::vertex_descriptor> vertices;

    auto connection_it = connections.find(edge_position);
    if (connection_it == connections.end()) return vertices;

    for (const auto& node_uid : connection_it->second)
    {
        auto vertex = graph.find_node(node_uid);
        if (vertex.has_value())
        {
            vertices.push_back(vertex.value());
            continue;
        }
        SPDLOG_INFO("Node with UID {} not found in graph.", std::string(node_uid));
    }
    return vertices;
}


/**
 * @brief Add edges of one entry of the edge vector to a graph.
 *
 * @param graph network graph to extend.
 * @param edge edge properties.
 * @param src_vertices source vertices.
 * @param dst_vertices target vertices.
 *
 * @details An edge is added for each pair of a source and a destination, because the bus does not tell which sender
 * belongs to which receiver.
 */
void add_edges_to_graph(
    NetworkGraph& graph, const Edge& edge, const std::vector<NetworkGraph::vertex_descriptor>& src_vertices,
    const std::vector<NetworkGraph::vertex_descriptor>& dst_vertices)
{
    for (const auto& src : src_vertices)
    {
        for (const auto& dst : dst_vertices)
        {
            Edge graph_edge = edge;
            if (!graph.graph_[src].is_invisible_) graph_edge.index_from_ = graph.graph_[src].uid_;
            if (!graph.graph_[dst].is_invisible_) graph_edge.index_to_ = graph.graph_[dst].uid_;
            graph.add_edge(src, dst, graph_edge);
        }
    }
}


/**
 * @brief Build network graph from collected nodes and edges.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph fill_graph(
    const std::vector<Node>& nodes, const std::vector<Edge>& edges, const EdgeConnections& node_src,
    const EdgeConnections& node_dst, bool is_dynamic)
{
    NetworkGraph graph;
    graph.is_dynamic_ = is_dynamic;

    for (const auto& node : nodes) graph.add_node(node);

    for (size_t edge_position = 0; edge_position < edges.size(); ++edge_position)
    {
        const Edge& edge = edges[edge_position];

        auto src_vertices = get_connected_vertices(graph, node_src, edge_position);
        auto dst_vertices = get_connected_vertices(graph, node_dst, edge_position);

        // A static graph shows known connections only: an edge without a source or without a destination is not
        // drawn at all, so no invisible node is added and nothing is written to the DOT text.
        if (!is_dynamic && (src_vertices.empty() || dst_vertices.empty()))
        {
            SPDLOG_INFO("Edge {} has no source or no destination, it is skipped.", std::string(edge.uid_));
            continue;
        }

        // An edge that is not on the bus at all has nothing to connect, it is not drawn either.
        if (src_vertices.empty() && dst_vertices.empty())
        {
            SPDLOG_INFO("Edge {} has no connections on the bus, it is skipped.", std::string(edge.uid_));
            continue;
        }

        if (src_vertices.empty())
        {
            SPDLOG_INFO("There is no source for this edge: {}", std::string(edge.uid_));
            src_vertices.push_back(graph.add_placeholder_node());
        }

        if (dst_vertices.empty())
        {
            SPDLOG_INFO("There is no destination for this edge: {}", std::string(edge.uid_));
            dst_vertices.push_back(graph.add_placeholder_node());
        }

        add_edges_to_graph(graph, edge, src_vertices, dst_vertices);
    }

    return graph;
}

}  // namespace knp::framework::visualizer::impl


namespace knp::framework::visualizer
{

/**
 * @brief Build network graph from a network.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Network& network)
{
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    impl::EdgeConnections node_src;
    impl::EdgeConnections node_dst;

    impl::get_static_connections(network, nodes, edges, node_src, node_dst);

    return impl::fill_graph(nodes, edges, node_src, node_dst, false);
}


/**
 * @brief Build network graph from a model.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Model& model)
{
    return build_graph(model.get_network());
}


/**
 * @brief Build network graph from a network and message bus subscriptions.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Network& network, knp::core::MessageEndpoint& endpoint)
{
    DynamicInfo info;
    info.bus_ = get_bus_connections(endpoint);
    return build_graph(network, info);
}


/**
 * @brief Build network graph from a model and message bus subscriptions.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Model& model, knp::core::MessageEndpoint& endpoint)
{
    return build_graph(model.get_network(), endpoint);
}


/**
 * @brief Build network graph from a network, bus subscriptions and entities known to the calling code.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Network& network, const DynamicInfo& info)
{
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    impl::EdgeConnections node_src;
    impl::EdgeConnections node_dst;

    impl::get_static_connections(network, nodes, edges, node_src, node_dst);

    // Connections described by the network are not used here: a loaded model rebuilds its projections, so the
    // presynaptic and postsynaptic UIDs of the network do not match the real connections. Sources and destinations
    // are taken from the bus, while the network gives names, sizes and types.
    node_src.clear();
    node_dst.clear();

    impl::get_dynamic_connections(info, nodes, edges, node_src, node_dst);

    return impl::fill_graph(nodes, edges, node_src, node_dst, true);
}


/**
 * @brief Build network graph from a model, bus subscriptions and entities known to the calling code.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Model& model, const DynamicInfo& info)
{
    return build_graph(model.get_network(), info);
}

}  // namespace knp::framework::visualizer
