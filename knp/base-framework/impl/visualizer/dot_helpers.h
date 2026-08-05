/**
 * @file dot_helpers.h
 * @brief Help functions for writing a dot file.
 * @kaspersky_support Kirill L.
 * @date 05.08.2026
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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
#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

extern "C"
{
#include <graphviz/gvc.h>
}


/**
 * @brief Write header and styles to DOT file
 *
 * @param out output file stream
 *
 * @details commentsed - for a connected graph
 */
void write_header_and_styles(std::ofstream& out);

/**
 * @brief Write edge to DOT file
 *
 * @param out output file stream
 * @param index_from source node UID
 * @param index_to target node UID
 * @param name edge name
 * @param size edge size
 * @param color edge color
 *
 * @details need to visualize dynamic
 */
template <typename T>
void write_edge_to_dot(
    std::ofstream& out, T index_from, T index_to, T name, T size, std::string const& color = "lightgray")
{
    // example of dot line:
    // "bdd469c7-66f2-43d9-b751-fb1e20732d3d" -> "12fe2450-0f03-4d93-a1e0-8d2f7a9277ae"
    // [label="Projection:\nfced41d5\nsize: 200", color="lightgray"];

    out << "    \"" << index_from << "\" -> \"" << index_to << "\" [label=\"Projection:\\n"
        << name << "\\nsize: " << size << "\", color=\"" << color << "\"];\n";
}

/**
 * @brief Write node to DOT file
 *
 * @param out output file stream
 * @param node_uid node UID
 * @param name node name
 * @param size node size
 * @param color node color
 * @param type_node node type
 *
 * @details need to visualize dynamic
 */
template <typename T>
void write_node_to_dot(
    std::ofstream& out, std::string const& node_uid, std::string const& name, T size,
    std::string const& color = "lightgray", std::string const& type_node = "Population")
{
    //examples:
    // "bdd469c7-66f2-43d9-b751-fb1e20732d3d" [label="Modificator:\nbdd469c7", color="yellow"];
    // "48bebcd2-80f6-488c-bbe8-0627cebc8af7" [label="Population:\nOUTPUT\nsize: 10", color="lightgray"];

    if constexpr (std::is_same_v<decltype(size), std::size_t>)
    {
        out << "    \"" << node_uid << "\" [label=\"" << type_node << ":\\n"
            << name << "\\nsize: " << size << "\", color=\"" << color << "\"];\n";
        return;
    }

    out << "    \"" << node_uid << "\" [label=\"" << type_node << ":\\n" << name << "\", color=\"" << color << "\"];\n";
}

/**
 * @brief Write edge without source to DOT file
 *
 * @param out output file stream
 * @param dst destination node UID
 * @param i counter for ghost node
 * @param edge_uid edge UID
 * @param size edge size
 * @param color edge color
 *
 * @details A graphviz can't draw an arrow from nowhere. It is necessary to make an invisible vertex.
 */
void write_edge_without_src(
    std::ofstream& out, knp::core::UID const& dst, int const& i, knp::core::UID const& edge_uid,
    std::string const& size, std::string const& color = "lightgray");

/**
 * @brief Write edge without destination to DOT file
 *
 * @param out output file stream
 * @param src source node UID
 * @param i counter for ghost node
 * @param edge_uid edge UID
 * @param size edge size
 * @param color edge color
 *
 * @details A graphviz can't draw an arrow to nowhere. It is necessary to make an invisible vertex.
 */
void write_edge_without_dst(
    std::ofstream& out, knp::core::UID const& src, int const& i, knp::core::UID const& edge_uid,
    std::string const& size, std::string const& color = "lightgray");

/**
 * @brief Convert DOT file to PNG image
 *
 * @param path_to_dot_file path to input DOT file
 * @param path_to_png_file path to output PNG file
 *
 * @return boolean indicating success
 */
bool convert_dot_to_png(const std::string& path_to_dot_file, const std::string& path_to_png_file);

/**
 * @brief Write bus messages (spike messages and synaptic impact messages) to DOT file
 *
 * @param spike_messages vector of spike message pairs
 * @param synaptic_messages vector of synaptic message pairs
 * @param graph network graph
 * @param out output file stream for DOT file
 */
void write_bus_messeges_to_dot(
    std::vector<std::pair<knp::core::UID, knp::core::UID>> const& spike_messages,
    std::vector<std::pair<knp::core::UID, knp::core::UID>> const& synaptic_messages,
    knp::framework::NetworkGraph const& graph, std::ofstream& out);

/**
 * @brief Write projections and populations to DOT file
 *
 * @param unique_nodes set of unique node UIDs
 * @param unique_edges set of unique edge UIDs
 * @param modificators set of modificator UIDs
 * @param out output file stream
 * @param dynamic_color color for dynamic elements
 * @param graph network graph
 * @param node_src map of node sources
 * @param node_dst map of node destinations
 */
void write_projections_and_populations_to_dot(
    std::set<knp::core::UID> const& unique_nodes, std::set<knp::core::UID> const& unique_edges,
    std::set<knp::core::UID>& modificators, std::ofstream& out, std::string& dynamic_color,
    knp::framework::NetworkGraph const& graph, std::map<knp::core::UID, knp::core::UID>& node_src,
    std::map<knp::core::UID, knp::core::UID>& node_dst);
