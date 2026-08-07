/**
 * @file generate_files.h
 * @brief Functions for generate dot/png files.
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
#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

extern "C"
{
#include <graphviz/gvc.h>
}


/**
 * @brief Create DOT file for static network.
 *
 * @param file_name output file name.
 * @param graph network graph.
 */
void create_dot_file_for_static_network(std::string& file_name, knp::framework::NetworkGraph& graph);


/**
 * @brief Create DOT file for bus messages.
 *
 * @param file_name output DOT file name.
 * @param graph network graph.
 * @param backend shared pointer to backend.
 */
void create_dot_file_for_bus(
    const std::string& file_name, const knp::framework::NetworkGraph& graph,
    std::shared_ptr<knp::core::Backend>& backend);


/**
 * @brief Create DOT file for dynamic network.
 *
 * @param path output path file name.
 * @param graph network graph.
 * @param backend shared pointer to backend.
 */
void create_dot_file_for_dynamic_network(
    const std::string& path, const knp::framework::NetworkGraph& graph, std::shared_ptr<knp::core::Backend>& backend);


/**
 * @brief Convert DOT file to PNG image.
 *
 * @param path_to_dot_file path to input DOT file.
 * @param path_to_png_file path to output PNG file.
 *
 * @return boolean indicating success.
 */
bool convert_dot_to_png(const std::string& path_to_dot_file, const std::string& path_to_png_file);


/**
 * @brief Create PNG file from DOT file.
 *
 * @param dot_file path to DOT file.
 * @param png_file path to PNG file.
 */
void create_png_file(const std::string& dot_file, const std::string& png_file);
