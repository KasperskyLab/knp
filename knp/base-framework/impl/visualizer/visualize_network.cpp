/**
 * @file visualize_network.cpp
 * @brief Text description of a network graph.
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

#include <knp/framework/visualizer/visualize_network.h>

#include <ostream>
#include <string>

#include "dot_helpers.h"


namespace knp::framework::visualizer::impl
{

/**
 * @brief Get description of a node as it is printed in a network description.
 *
 * @param graph graph storage.
 * @param vertex vertex descriptor.
 * @param is_source flag showing that the node is a source of an edge.
 *
 * @return node kind and name, or a placeholder for an invisible node.
 *
 * @details The kind is written for a neighbour as well, so a WTA handler or a channel is recognizable everywhere in
 * the description, not only in its own line. A placeholder does not repeat a node name, so it is not mixed up with a
 * population called "INPUT" or "OUTPUT". An invisible node exists in a dynamic graph only.
 */
std::string get_description_name(const BaseGraph& graph, BaseGraph::vertex_descriptor vertex, bool is_source)
{
    if (graph[vertex].is_invisible()) return is_source ? "NO_SOURCE" : "NO_DESTINATION";
    return graph[vertex].get_node_type() + " '" + graph[vertex].get_name() + "'";
}


/**
 * @brief Get description of an edge as it is printed in a network description.
 *
 * @param edge edge to describe.
 *
 * @return edge kind, and its name when the edge has one.
 */
std::string get_description_title(const Edge& edge)
{
    if (edge.get_name().empty()) return edge.get_edge_type();
    return edge.get_edge_type() + ": " + edge.get_name();
}

}  // namespace knp::framework::visualizer::impl


namespace knp::framework::visualizer
{

/**
 * @brief Print node and edge connections of a network graph.
 *
 * @details Parameters and return value are described in the header.
 */
void print_network_description(const NetworkGraph& graph, std::ostream& out)
{
    const BaseGraph& network_graph = graph.get_graph();

    // The description shows exactly what is drawn: the same nodes and the same edges as the DOT text.
    for (auto [vertex_it, vertex_end] = boost::vertices(network_graph); vertex_it != vertex_end; ++vertex_it)
    {
        if (!impl::is_node_drawn(network_graph, *vertex_it)) continue;

        const Node& node = network_graph[*vertex_it];
        out << "    " << node.get_node_type() << " #" << *vertex_it << " '" << node.get_name() << "'";
        if (node.get_size().has_value()) out << " of size " << node.get_size().value();

        out << ": receive from";
        for (auto [edge_it, edge_end] = boost::in_edges(*vertex_it, network_graph); edge_it != edge_end; ++edge_it)
        {
            if (!impl::is_edge_drawn(graph, *edge_it)) continue;
            auto source_id = boost::source(*edge_it, network_graph);
            out << " #" << impl::get_description_name(network_graph, source_id, true) << " ("
                << impl::get_description_title(network_graph[*edge_it]) << ");";
        }

        out << " send to";
        for (auto [edge_it, edge_end] = boost::out_edges(*vertex_it, network_graph); edge_it != edge_end; ++edge_it)
        {
            if (!impl::is_edge_drawn(graph, *edge_it)) continue;
            auto target_id = boost::target(*edge_it, network_graph);
            out << " #" << impl::get_description_name(network_graph, target_id, false) << " ("
                << impl::get_description_title(network_graph[*edge_it]) << ");";
        }

        out << '\n';
    }
}

}  // namespace knp::framework::visualizer
