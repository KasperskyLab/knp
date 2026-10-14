/**
 * @file graph_builder.cpp
 * @brief Building a Boost Graph from collected nodes and edges.
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


namespace
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
std::vector<knp::framework::NetworkGraph::vertex_descriptor> get_connected_vertices(
    const knp::framework::NetworkGraph& graph, const EdgeConnections& connections, size_t edge_position)
{
    std::vector<knp::framework::NetworkGraph::vertex_descriptor> vertices;

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
    knp::framework::NetworkGraph& graph, const knp::framework::Edge& edge,
    const std::vector<knp::framework::NetworkGraph::vertex_descriptor>& src_vertices,
    const std::vector<knp::framework::NetworkGraph::vertex_descriptor>& dst_vertices)
{
    for (const auto& src : src_vertices)
    {
        for (const auto& dst : dst_vertices)
        {
            knp::framework::Edge graph_edge = edge;
            if (!graph.graph_[src].is_invisible_) graph_edge.index_from_ = graph.graph_[src].uid_;
            if (!graph.graph_[dst].is_invisible_) graph_edge.index_to_ = graph.graph_[dst].uid_;
            graph.add_edge(src, dst, graph_edge);
        }
    }
}

}  // namespace


/**
 * @brief Build network graph from collected nodes and edges.
 *
 * @details Parameters and return value are described in the header.
 */
knp::framework::NetworkGraph fill_graph(
    const std::vector<knp::framework::Node>& nodes, const std::vector<knp::framework::Edge>& edges,
    const EdgeConnections& node_src, const EdgeConnections& node_dst, bool is_dynamic)
{
    knp::framework::NetworkGraph graph;
    graph.is_dynamic_ = is_dynamic;

    for (const auto& node : nodes) graph.add_node(node);

    for (size_t edge_position = 0; edge_position < edges.size(); ++edge_position)
    {
        const knp::framework::Edge& edge = edges[edge_position];

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
