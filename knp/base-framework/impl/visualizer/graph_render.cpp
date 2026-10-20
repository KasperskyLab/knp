/**
 * @file graph_render.cpp
 * @brief Rendering a network graph with graphviz.
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

#include <knp/framework/visualizer/graph_visualization.h>

#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

extern "C"
{
#include <graphviz/gvc.h>
}


namespace knp::framework::visualizer::impl
{

/**
 * @brief Get type of the length parameter of the gvRenderData() function.
 *
 * @tparam Length type of the length parameter.
 *
 * @return value of the length type. The function is used in an unevaluated context only, so it has no definition.
 *
 * @details Graphviz 13 changed the length parameter of gvRenderData() from unsigned int to size_t. Numeric version
 * macros appeared in Graphviz 14 only, so the type is taken from the function declaration instead of a version check.
 */
template <typename Length>
Length get_render_length_type(int (*)(GVC_t*, graph_t*, const char*, char**, Length*));


/**
 * @brief Type of the length parameter of the gvRenderData() function in the installed Graphviz.
 */
using RenderLength = decltype(get_render_length_type(&gvRenderData));


/**
 * @brief Render DOT text to a stream.
 *
 * @param dot_text DOT text of a graph.
 * @param out output stream for the rendered image.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @return boolean indicating success.
 *
 * @details The DOT text is parsed from memory and the image is rendered to memory, so no intermediate file is
 * created. The layout engine is taken from the `layout` attribute of the graph.
 */
bool render_dot_to_stream(const std::string& dot_text, std::ostream& out, const std::string& format)
{
    // 1. Initialize the Graphviz context.
    auto gvc = std::unique_ptr<GVC_t, decltype(&gvFreeContext)>(gvContext(), gvFreeContext);
    if (!gvc)
    {
        SPDLOG_ERROR("Error: couldn't create the graphviz context.");
        return false;
    }

    // 2. Read the DOT text from memory. A copy is made because agmemread() takes a non-constant buffer.
    std::string dot_buffer = dot_text;
    auto graph = std::unique_ptr<Agraph_t, decltype(&agclose)>(agmemread(dot_buffer.data()), agclose);
    if (!graph)
    {
        SPDLOG_ERROR("Error: couldn't read the generated DOT text.");
        return false;
    }

    // 3. Compute the layout using the engine from the "layout" attribute.
    if (gvLayoutJobs(gvc.get(), graph.get()) != 0)
    {
        SPDLOG_ERROR("Layout computation failed.");
        return false;
    }

    // 4. Render the layout to memory.
    char* rendered_data = nullptr;
    RenderLength rendered_length = 0;
    const int render_result = gvRenderData(gvc.get(), graph.get(), format.c_str(), &rendered_data, &rendered_length);

    // 5. Write the rendered image to the stream and free the graphviz buffers.
    bool is_written = false;
    if (render_result == 0 && rendered_data != nullptr)
    {
        out.write(rendered_data, static_cast<std::streamsize>(rendered_length));
        is_written = out.good();
        if (!is_written) SPDLOG_ERROR("Failed to write the rendered image to the stream.");
    }
    else
    {
        SPDLOG_ERROR("Rendering error for the \"{}\" format.", format);
    }

    gvFreeRenderData(rendered_data);
    gvFreeLayout(gvc.get(), graph.get());

    return is_written;
}

}  // namespace knp::framework::visualizer::impl


namespace knp::framework::visualizer
{

/**
 * @brief Render network graph to a stream as an image.
 *
 * @details Parameters and return value are described in the header.
 */
bool render_graph(const NetworkGraph& graph, std::ostream& out, const std::string& format)
{
    std::ostringstream dot_stream;
    write_dot(graph, dot_stream);
    return impl::render_dot_to_stream(dot_stream.str(), out, format);
}


/**
 * @brief Save network graph to an image file.
 *
 * @details Parameters are described in the header.
 */
void save_image(const NetworkGraph& graph, const std::filesystem::path& file, const std::string& format)
{
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());

    std::ofstream out(file, std::ios::binary);
    if (!render_graph(graph, out, format))
        throw std::runtime_error("Failed to render the graph to file " + file.string() + ".");

    SPDLOG_INFO("The {} file is saved in: {}", format, std::filesystem::absolute(file).string());
}

}  // namespace knp::framework::visualizer
