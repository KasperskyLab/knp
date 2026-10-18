/**
 * @file render_helpers.h
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

#pragma once

#include <ostream>
#include <string>

extern "C"
{
#include <gvc.h>
}


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
bool render_dot_to_stream(const std::string& dot_text, std::ostream& out, const std::string& format);
