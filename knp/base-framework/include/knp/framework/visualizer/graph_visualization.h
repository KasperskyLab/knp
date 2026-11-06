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
 * @brief Print a human-readable description of a network graph.
 *
 * @details The description lists each visible node together with its incoming and outgoing
 * edges, using the same naming and ordering as the DOT export. This guarantees
 * that the textual description and the visual picture always match.
 *
 * @param graph `NetworkGraph` to describe.
 * @param out output stream; defaults to `std::cout`.
 *
 * @note The function does not modify the graph.
 */
KNP_DECLSPEC void print_network_description(const NetworkGraph& graph, std::ostream& out = std::cout);


/**
 * @brief Export a network graph to DOT format.
 *
 * @param graph `NetworkGraph` to export.
 * @param out output stream.
 *
 * @details The generated DOT text contains a header, global style settings,
 * node definitions, edge definitions and a closing brace.
 */
KNP_DECLSPEC void write_dot(const NetworkGraph& graph, std::ostream& out);


/**
 * @brief Export a network graph to a DOT file.
 *
 * @param graph `NetworkGraph` to export.
 * @param file path to the output file. Parent directories are created if they do not exist.
 *
 * @throw std::filesystem::filesystem_error If a directory cannot be created.
 * @throw std::runtime_error If the file cannot be opened or written.
 *
 * @note The function writes the same content as the stream overload.
 */
KNP_DECLSPEC void write_dot(const NetworkGraph& graph, const std::filesystem::path& file);


/**
 * @brief Render a network graph to an image stream using Graphviz.
 *
 * @param graph `NetworkGraph` to render.
 * @param out output stream - binary mode for PNG/SVG/etc.
 * @param format graphviz output format, for example, "png", "svg", "pdf". Default is "png".
 *
 * @return `true` on successful rendering, `false` otherwise.
 *
 * @details The function generates DOT text internally, feeds it to Graphviz and streams the
 * rendered image back to the caller. No temporary files are created.
 */
KNP_DECLSPEC bool render_graph(const NetworkGraph& graph, std::ostream& out, const std::string& format = "png");


/**
 * @brief Render a network graph directly to an image file.
 *
 * @param graph `NetworkGraph` to render.
 * @param file target file path. Parent directories are created if needed.
 * @param format graphviz output format, for example, "png", "svg", "pdf". Default is "png".
 *
 * @throw std::filesystem::filesystem_error If a directory cannot be created.
 * @throw std::runtime_error If rendering fails or the file cannot be written.
 */
KNP_DECLSPEC void render_graph(
    const NetworkGraph& graph, const std::filesystem::path& file, const std::string& format = "png");

}  // namespace knp::framework::visualizer
