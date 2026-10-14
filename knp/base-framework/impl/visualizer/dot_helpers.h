/**
 * @file dot_helpers.h
 * @brief Help functions for writing a dot text.
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

#include <knp/framework/visualizer/visualize_network.h>

#include <cstddef>
#include <ostream>
#include <string>


/**
 * @brief Color of a static element.
 */
inline const std::string static_color = "lightgray";

/**
 * @brief Color of an element that is known from the message bus.
 */
inline const std::string dynamic_color = "yellow";


/**
 * @brief Check that a node is drawn.
 *
 * @param graph graph storage.
 * @param vertex vertex descriptor.
 *
 * @return boolean indicating that the node is visible on a picture.
 */
bool is_node_drawn(const knp::framework::BaseGraph& graph, knp::framework::BaseGraph::vertex_descriptor vertex);


/**
 * @brief Check that an edge is drawn.
 *
 * @param graph network graph.
 * @param edge edge descriptor.
 *
 * @return boolean indicating that the edge is written to the DOT text.
 *
 * @details An edge with an invisible node belongs to a dynamic graph only, a static graph shows known connections
 * only. The same check is used by a network description, so a picture and a description always match.
 */
bool is_edge_drawn(const knp::framework::NetworkGraph& graph, const knp::framework::BaseGraph::edge_descriptor& edge);


/**
 * @brief Escape a text that is written to a DOT attribute value.
 *
 * @param text source text.
 *
 * @return text with escaped quotes and backslashes.
 */
std::string escape_dot_text(const std::string& text);


/**
 * @brief Write header and styles to DOT text.
 *
 * @param out output stream.
 *
 * @details Settings are made for a graph with unconnected parts.
 */
void write_header_and_styles(std::ostream& out);


/**
 * @brief Write node to DOT text.
 *
 * @param out output stream.
 * @param node_id node identifier in the DOT text.
 * @param node node to write.
 * @param color node color.
 *
 * @details Example of a DOT line:
 * `1 [label="Population:\nOUTPUT\nsize: 10", color="lightgray"];`
 */
void write_node_to_dot(
    std::ostream& out, size_t node_id, const knp::framework::Node& node, const std::string& color = static_color);


/**
 * @brief Write invisible node to DOT text.
 *
 * @param out output stream.
 * @param node_id node identifier in the DOT text.
 *
 * @details A graphviz can't draw an arrow from nowhere. It is necessary to make an invisible vertex.
 */
void write_invisible_node_to_dot(std::ostream& out, size_t node_id);


/**
 * @brief Write edge to DOT text.
 *
 * @param out output stream.
 * @param index_from source node identifier in the DOT text.
 * @param index_to target node identifier in the DOT text.
 * @param edge edge to write.
 * @param color edge color.
 * @param style edge style.
 *
 * @details Example of a DOT line:
 * `0 -> 1 [label="Projection:\nfced41d5\nsize: 200", color="lightgray", style="solid"];`
 */
void write_edge_to_dot(
    std::ostream& out, size_t index_from, size_t index_to, const knp::framework::Edge& edge,
    const std::string& color = static_color, const std::string& style = "solid");


/**
 * @brief Write whole network graph to DOT text.
 *
 * @param out output stream.
 * @param graph network graph.
 *
 * @details Colors of dynamic elements are written for a graph built with bus subscriptions only, so a static graph
 * contains static information only and has no dotted edges.
 */
void write_graph_to_dot(std::ostream& out, const knp::framework::NetworkGraph& graph);
