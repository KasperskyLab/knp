/**
 * @file dot_writer.cpp
 * @brief Writing a network graph in DOT format.
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

#include "dot_writer.h"

#include <knp/framework/visualizer/graph_visualization.h>

#include <spdlog/spdlog.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>


namespace knp::framework::visualizer::impl
{

/**
 * @brief Check that a node is known from the message bus or works as a dynamic entity.
 *
 * @param node node to check.
 *
 * @return boolean indicating that the node is not an ordinary population.
 */
bool is_dynamic_node(const Node& node)
{
    return node.get_node_type() != population_node_type;
}


/**
 * @brief Get color of a node.
 *
 * @param node node to draw.
 * @param is_dynamic flag showing that the graph was built with bus subscriptions.
 *
 * @return node color.
 */
std::string get_node_color(const Node& node, bool is_dynamic)
{
    if (!is_dynamic) return static_color;
    return is_dynamic_node(node) ? dynamic_color : static_color;
}


/**
 * @brief Get color of an edge.
 *
 * @param src_node source node of the edge.
 * @param edge edge to draw.
 * @param is_dynamic flag showing that the graph was built with bus subscriptions.
 *
 * @return edge color.
 *
 * @details An edge is dynamic if its source is dynamic or if the edge itself is not described by the network.
 */
std::string get_edge_color(const Node& src_node, const Edge& edge, bool is_dynamic)
{
    if (!is_dynamic) return static_color;
    if (is_dynamic_node(src_node)) return dynamic_color;
    return edge.is_static() ? static_color : dynamic_color;
}


/**
 * @brief Get style of an edge.
 *
 * @param has_invisible_node flag showing that the edge has no source or no destination.
 *
 * @return edge style.
 *
 * @details A dotted edge has an invisible node on one side. Such an edge is drawn in a dynamic graph only, because
 * a static graph skips an edge with an unknown source or destination.
 */
std::string get_edge_style(bool has_invisible_node)
{
    return has_invisible_node ? "dotted" : "solid";
}


/**
 * @brief Check that a node is drawn.
 *
 * @details Parameters and return value are described in the header.
 */
bool is_node_drawn(const BaseGraph& graph, BaseGraph::vertex_descriptor vertex)
{
    return !graph[vertex].is_invisible();
}


/**
 * @brief Check that an edge is drawn, so that a picture and a description always match.
 *
 * @details Parameters and return value are described in the header.
 */
bool is_edge_drawn(const NetworkGraphImpl& graph, const BaseGraph::edge_descriptor& edge)
{
    const BaseGraph& network_graph = graph.get_graph();
    const bool has_invisible_node = network_graph[boost::source(edge, network_graph)].is_invisible() ||
                                    network_graph[boost::target(edge, network_graph)].is_invisible();
    return graph.is_dynamic() || !has_invisible_node;
}


/**
 * @brief Escape a text that is written to a DOT attribute value.
 *
 * @details Parameters and return value are described in the header.
 */
std::string escape_dot_text(const std::string& text)
{
    std::string result;
    result.reserve(text.size());
    for (char symbol : text)
    {
        // A control character is skipped.
        if (0 != std::iscntrl(static_cast<unsigned char>(symbol))) continue;
        if ('"' == symbol || '\\' == symbol) result.push_back('\\');
        result.push_back(symbol);
    }
    return result;
}


/**
 * @brief Write header and styles to DOT text.
 *
 * @details Parameters and return value are described in the header.
 */
void write_header_and_styles(std::ostream& out)
{
    // Digraph - directed.
    out << "digraph G {\n";

    // For unconnected graphs this is needed.
    out << "    layout=neato;\n";          // Change engine to neato (physical force model).
    out << "    pack=true;\n";             // Enable compact packing of independent components.
    out << "    packmode=\"graph\";\n\n";  // Pack each subgroup as a separate mini-graph.

    // Global design settings for the graph.
    out << "    // Canvas settings.\n";
    out << "    bgcolor=\"#FAFAFA\";\n";  // Light gray soft background page.
    out << "    splines=true;\n";         // Beautiful smooth curves for arrows.
    out << "    overlap=false;\n";
    out << "    sep=\"+30\";\n\n";

    out << "    // Default node styling.\n";
    out << "    node [\n";
    out << "        fontname=\"Helvetica,Arial,sans-serif\",\n";  // Modern font.
    out << "        fontsize=11,\n";
    out << "        shape=box,\n";                 // Rectangular shape.
    out << "        style=\"filled,rounded\",\n";  // Fill with color + rounded corners.
    out << "        penwidth=0,\n";                // Remove rigid black border around node.
    out << "        margin=\"0.2,0.1\"\n";         // Internal padding for text.
    out << "    ];\n\n";

    out << "    // Default edge (arrow) styling.\n";
    out << "    edge [\n";
    out << "        fontname=\"Helvetica,Arial,sans-serif\",\n";
    out << "        fontsize=9,\n";
    out << "        fontcolor=\"#555555\",\n";  // Soft gray color for text above arrows.
    out << "        color=\"#B0BEC5\",\n";      // Beautiful steel color for the arrow itself.
    out << "        penwidth=1.5,\n";           // Make line slightly thicker than default.
    out << "        arrowsize=0.8\n";           // Slightly reduce arrowhead size.
    out << "    ];\n\n";
}


/**
 * @brief Write node to DOT text.
 *
 * @details Parameters and return value are described in the header.
 */
void write_node_to_dot(size_t node_id, const Node& node, std::ostream& out, const std::string& color)
{
    out << "    " << node_id << " [label=\"" << escape_dot_text(node.get_node_type()) << ":\\n"
        << escape_dot_text(node.get_name());
    if (node.get_size().has_value()) out << "\\nsize: " << node.get_size().value();
    out << "\", color=\"" << color << "\"];\n";
}


/**
 * @brief Write invisible node to DOT text.
 *
 * @details Parameters and return value are described in the header.
 */
void write_invisible_node_to_dot(size_t node_id, std::ostream& out)
{
    out << "    " << node_id << " [shape=point, style=invis];\n";
}


/**
 * @brief Write edge to DOT text.
 *
 * @details Parameters and return value are described in the header.
 */
void write_edge_to_dot(
    size_t index_from, size_t index_to, const Edge& edge, std::ostream& out, const std::string& color,
    const std::string& style)
{
    out << "    " << index_from << " -> " << index_to << " [label=\"" << escape_dot_text(edge.get_edge_type());
    // An edge without a name is labelled with its kind only.
    if (!edge.get_name().empty()) out << ":\\n" << escape_dot_text(edge.get_name());
    if (edge.get_size().has_value()) out << "\\nsize: " << edge.get_size().value();
    out << "\", color=\"" << color << "\", style=\"" << style << "\"];\n";
}


/**
 * @brief Write whole network graph to DOT text.
 *
 * @details Parameters and return value are described in the header.
 */
void write_graph_to_dot(const NetworkGraphImpl& graph, std::ostream& out)
{
    const BaseGraph& network_graph = graph.get_graph();

    write_header_and_styles(out);

    if (boost::num_vertices(network_graph) == 0)
    {
        SPDLOG_WARN("No nodes or edges to visualize.");
        out << "}\n";
        return;
    }

    // Write nodes to dot. A vertex descriptor is an index, so it is unique and can be used as a node identifier.
    for (auto [vertex_it, vertex_end] = boost::vertices(network_graph); vertex_it != vertex_end; ++vertex_it)
    {
        const auto node_id = static_cast<size_t>(*vertex_it);
        const Node& node = network_graph[*vertex_it];

        if (!is_node_drawn(network_graph, *vertex_it))
        {
            // A static graph never contains invisible nodes and dotted edges.
            if (graph.is_dynamic()) write_invisible_node_to_dot(node_id, out);
            continue;
        }
        write_node_to_dot(node_id, node, out, get_node_color(node, graph.is_dynamic()));
    }

    // Write edges to dot.
    for (auto [edge_it, edge_end] = boost::edges(network_graph); edge_it != edge_end; ++edge_it)
    {
        const auto source_id = boost::source(*edge_it, network_graph);
        const auto target_id = boost::target(*edge_it, network_graph);
        const Edge& edge = network_graph[*edge_it];

        if (!is_edge_drawn(graph, *edge_it)) continue;

        const bool has_invisible_node =
            network_graph[source_id].is_invisible() || network_graph[target_id].is_invisible();

        write_edge_to_dot(
            static_cast<size_t>(source_id), static_cast<size_t>(target_id), edge, out,
            get_edge_color(network_graph[source_id], edge, graph.is_dynamic()), get_edge_style(has_invisible_node));
    }

    out << "}\n";
}

}  // namespace knp::framework::visualizer::impl


namespace knp::framework::visualizer
{

/**
 * @brief Write network graph to a stream in DOT format.
 *
 * @details Parameters and return value are described in the header.
 */
void write_dot(const NetworkGraph& graph, std::ostream& out)
{
    impl::write_graph_to_dot(graph.get_impl(), out);
}


/**
 * @brief Write network graph to a file in DOT format.
 *
 * @details Parameters are described in the header.
 */
void write_dot(const NetworkGraph& graph, const std::filesystem::path& file)
{
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());

    std::ofstream out(file);
    write_dot(graph, out);
    if (!out.good()) throw std::runtime_error("Failed to write to file " + file.string() + ".");

    SPDLOG_INFO("The DOT file is saved in: {}", std::filesystem::absolute(file).string());
}

}  // namespace knp::framework::visualizer
