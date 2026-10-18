/**
 * @file visualize.cpp
 * @brief Saving DOT and image files of a network graph.
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

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "render_helpers.h"


namespace knp::framework::visualizer
{

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

    std::filesystem::create_directories(dot_dir);
    std::filesystem::create_directories(image_dir);

    std::ostringstream dot_stream;
    write_dot(graph, dot_stream);
    const std::string dot_text = dot_stream.str();

    const auto dot_file = dot_dir / (name + ".dot");
    {
        std::ofstream out(dot_file);
        out << dot_text;
        if (!out.good()) throw std::runtime_error("Failed to write to file " + dot_file.string() + ".");
    }
    SPDLOG_INFO("The DOT file is saved in: {}", std::filesystem::absolute(dot_file).string());

    const auto image_file = image_dir / (name + "." + format);
    {
        std::ofstream out(image_file, std::ios::binary);
        if (!impl::render_dot_to_stream(dot_text, out, format))
            throw std::runtime_error("Failed to render the graph to file " + image_file.string() + ".");
    }
    SPDLOG_INFO("The {} file is saved in: {}", format, std::filesystem::absolute(image_file).string());

    return image_file;
}

}  // namespace knp::framework::visualizer
