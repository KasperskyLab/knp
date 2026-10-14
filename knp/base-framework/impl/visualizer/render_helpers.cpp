/**
 * @file render_helpers.cpp
 * @brief Functions for rendering a dot text with graphviz.
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

#include "render_helpers.h"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <memory>
#include <string>


/**
 * @brief Render DOT text to a stream.
 *
 * @details Parameters and return value are described in the header.
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
    // Graphviz changed the length parameter of gvRenderData() from unsigned int to size_t in version 9. The headers
    // provide no numeric version macro, so the old type is selected by KNP_GRAPHVIZ_LEGACY_RENDER_LENGTH, which the
    // build system defines for Graphviz 8 and older.
#ifdef KNP_GRAPHVIZ_LEGACY_RENDER_LENGTH
    unsigned int rendered_length = 0;
#else
    size_t rendered_length = 0;
#endif
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
