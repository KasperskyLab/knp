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

#include <knp/core/uid.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <cstddef>
#include <map>
#include <set>
#include <vector>


/**
 * @brief Connections of edges: edge position in the edge vector to a set of node UIDs.
 *
 * @details The key is a position and not a UID, because an edge can have no meaningful UID of its own, and several
 * entities can have the same UID. A set is needed because one projection can have several senders or several
 * receivers on the bus.
 */
using EdgeConnections = std::map<size_t, std::set<knp::core::UID>>;


/**
 * @brief Positions of entities in a node or an edge vector, by UID.
 */
using UidIndex = std::map<knp::core::UID, size_t>;


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
knp::framework::NetworkGraph fill_graph(
    const std::vector<knp::framework::Node>& nodes, const std::vector<knp::framework::Edge>& edges,
    const EdgeConnections& node_src, const EdgeConnections& node_dst, bool is_dynamic);
