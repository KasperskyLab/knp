/**
 * @file visualize_network.cpp
 * @brief Network graph building, description and visualization.
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

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "dot_helpers.h"
#include "graph_helpers.h"
#include "render_helpers.h"


namespace knp::framework
{

namespace
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
    if (graph[vertex].is_invisible_) return is_source ? "NO_SOURCE" : "NO_DESTINATION";
    return fmt::format("{} '{}'", graph[vertex].node_type_, graph[vertex].name_);
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
    if (edge.name_.empty()) return edge.edge_type_;
    return fmt::format("{}: {}", edge.edge_type_, edge.name_);
}

}  // namespace


/**
 * @brief Build network graph from a network.
 *
 * @details Parameters and return value are described in the header.
 */
NetworkGraph build_graph(const knp::framework::Network& network)
{
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    EdgeConnections node_src;
    EdgeConnections node_dst;

    get_static_connections(network, nodes, edges, node_src, node_dst);

    return fill_graph(nodes, edges, node_src, node_dst, false);
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
    EdgeConnections node_src;
    EdgeConnections node_dst;

    get_static_connections(network, nodes, edges, node_src, node_dst);

    // Connections described by the network are not used here: a loaded model rebuilds its projections, so the
    // presynaptic and postsynaptic UIDs of the network do not match the real connections. Sources and destinations
    // are taken from the bus, while the network gives names, sizes and types.
    node_src.clear();
    node_dst.clear();

    get_dynamic_connections(info, nodes, edges, node_src, node_dst);

    return fill_graph(nodes, edges, node_src, node_dst, true);
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


/**
 * @brief Print node and edge connections of a network graph.
 *
 * @details Parameters and return value are described in the header.
 */
void print_network_description(const NetworkGraph& graph)
{
    const BaseGraph& network_graph = graph.graph_;

    // Summary line. The counters go over the drawn nodes and edges only, so they can be compared with the picture.
    // size_t node_count = 0;
    // for (auto [vertex_it, vertex_end] = boost::vertices(network_graph); vertex_it != vertex_end; ++vertex_it)
    // {
    //     if (is_node_drawn(network_graph, *vertex_it)) ++node_count;
    // }
    //
    // size_t edge_count = 0;
    // for (auto [edge_it, edge_end] = boost::edges(network_graph); edge_it != edge_end; ++edge_it)
    // {
    //     if (is_edge_drawn(graph, *edge_it)) ++edge_count;
    // }
    //
    // SPDLOG_INFO(
    //     "{} network graph: {} nodes, {} edges.", graph.is_dynamic_ ? "Dynamic" : "Static", node_count, edge_count);

    // The description shows exactly what is drawn: the same nodes and the same edges as the DOT text.
    for (auto [vertex_it, vertex_end] = boost::vertices(network_graph); vertex_it != vertex_end; ++vertex_it)
    {
        if (!is_node_drawn(network_graph, *vertex_it)) continue;

        const Node& node = network_graph[*vertex_it];
        std::string log_msg = fmt::format("    {} #{} '{}'", node.node_type_, *vertex_it, node.name_);
        if (node.size_.has_value()) log_msg += fmt::format(" of size {}", node.size_.value());

        log_msg += ": receive from";
        for (auto [edge_it, edge_end] = boost::in_edges(*vertex_it, network_graph); edge_it != edge_end; ++edge_it)
        {
            if (!is_edge_drawn(graph, *edge_it)) continue;
            auto source_id = boost::source(*edge_it, network_graph);
            log_msg += fmt::format(
                " #{} ({});", get_description_name(network_graph, source_id, true),
                get_description_title(network_graph[*edge_it]));
        }

        log_msg += " send to";
        for (auto [edge_it, edge_end] = boost::out_edges(*vertex_it, network_graph); edge_it != edge_end; ++edge_it)
        {
            if (!is_edge_drawn(graph, *edge_it)) continue;
            auto target_id = boost::target(*edge_it, network_graph);
            log_msg += fmt::format(
                " #{} ({});", get_description_name(network_graph, target_id, false),
                get_description_title(network_graph[*edge_it]));
        }

        SPDLOG_INFO("{}", log_msg);
    }
}


/**
 * @brief Write network graph to a stream in DOT format.
 *
 * @details Parameters and return value are described in the header.
 */
void write_dot(const NetworkGraph& graph, std::ostream& out)
{
    write_graph_to_dot(out, graph);
}


/**
 * @brief Render network graph to a stream as an image.
 *
 * @details Parameters and return value are described in the header.
 */
bool render_graph(const NetworkGraph& graph, std::ostream& out, const std::string& format)
{
    std::ostringstream dot_stream;
    write_dot(graph, dot_stream);
    return render_dot_to_stream(dot_stream.str(), out, format);
}


/**
 * @brief Save DOT and image files of a network graph.
 *
 * @details Parameters and return value are described in the header.
 */
std::filesystem::path visualize(
    const NetworkGraph& graph, const std::string& name, const std::filesystem::path& dir, const std::string& format)
{
    const auto dot_dir = dir / "dot_files";
    const auto image_dir = dir / (format + "_files");

    std::error_code error_code;
    std::filesystem::create_directories(dot_dir, error_code);
    std::filesystem::create_directories(image_dir, error_code);
    if (error_code)
    {
        SPDLOG_ERROR("Failed to create directories: {}", error_code.message());
        return {};
    }

    std::ostringstream dot_stream;
    write_dot(graph, dot_stream);
    const std::string dot_text = dot_stream.str();

    const auto dot_file = dot_dir / (name + ".dot");
    {
        std::ofstream out(dot_file);
        out << dot_text;
        if (!out.good())
        {
            SPDLOG_ERROR("Failed to write to file {}", dot_file.string());
            return {};
        }
    }
    SPDLOG_INFO("The DOT file is saved in: {}", std::filesystem::absolute(dot_file).string());

    const auto image_file = image_dir / (name + "." + format);
    {
        std::ofstream out(image_file, std::ios::binary);
        if (!render_dot_to_stream(dot_text, out, format))
        {
            SPDLOG_ERROR("The {} file is not saved.", format);
            return {};
        }
    }
    SPDLOG_INFO("The {} file is saved in: {}", format, std::filesystem::absolute(image_file).string());

    return image_file;
}

}  // namespace knp::framework
