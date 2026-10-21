/**
 * @file graph_visualization.h
 * @brief Network graph description and visualization.
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

#include <knp/framework/visualizer/network_graph.h>

#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Print node and edge connections of a network graph.
 *
 * @param graph network graph.
 * @param out output stream, standard output by default.
 *
 * @details The function writes a line for each node and its incoming and outgoing edges. The same nodes and edges are
 * written as in the DOT text, so a description and a picture always match.
 */
KNP_DECLSPEC void print_network_description(const NetworkGraph& graph, std::ostream& out = std::cout);


/**
 * @brief Write network graph to a stream in DOT format.
 *
 * @param graph network graph.
 * @param out output stream.
 */
KNP_DECLSPEC void write_dot(const NetworkGraph& graph, std::ostream& out);


/**
 * @brief Write network graph to a file in DOT format.
 *
 * @param graph network graph.
 * @param file path to the DOT file. A parent directory is created if it does not exist.
 *
 * @throw std::filesystem::filesystem_error if a directory cannot be created.
 * @throw std::runtime_error if the file cannot be written.
 */
KNP_DECLSPEC void write_dot(const NetworkGraph& graph, const std::filesystem::path& file);


/**
 * @brief Render network graph to a stream as an image.
 *
 * @param graph network graph.
 * @param out output stream.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @return boolean indicating success.
 *
 * @details No intermediate file is created: the DOT text and the rendered image are kept in memory.
 */
KNP_DECLSPEC bool render_graph(const NetworkGraph& graph, std::ostream& out, const std::string& format = "png");


/**
 * @brief Render network graph to a file as an image.
 *
 * @param graph network graph.
 * @param file path to the image file. A parent directory is created if it does not exist.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @throw std::filesystem::filesystem_error if a directory cannot be created.
 * @throw std::runtime_error if the graph cannot be rendered or the file cannot be written.
 */
KNP_DECLSPEC void render_graph(
    const NetworkGraph& graph, const std::filesystem::path& file, const std::string& format = "png");

}  // namespace knp::framework::visualizer
