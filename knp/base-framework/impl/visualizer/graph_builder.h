/**
 * @file graph_builder.h
 * @brief Building a Boost Graph from collected nodes and edges.
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

#include <vector>

#include "graph_connections.h"


namespace knp::framework::visualizer::impl
{

/**
 * @brief Build network graph from collected nodes and edges.
 *
 * @param nodes vector of nodes.
 * @param edges vector of edges.
 * @param node_src map of edge sources.
 * @param node_dst map of edge destinations.
 * @param is_dynamic flag showing that bus subscriptions were used.
 *
 * @return network graph.
 *
 * @details An edge with several sources or several destinations is added once for each pair. In a static graph an
 * edge without a source or without a destination is skipped. In a dynamic graph such an edge gets an invisible node
 * on that side, and an edge without any connections is skipped.
 */
NetworkGraph fill_graph(
    const std::vector<Node>& nodes, const std::vector<Edge>& edges, const EdgeConnections& node_src,
    const EdgeConnections& node_dst, bool is_dynamic);

}  // namespace knp::framework::visualizer::impl
