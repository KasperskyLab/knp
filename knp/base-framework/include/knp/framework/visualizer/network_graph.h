/**
 * @file network_graph.h
 * @brief Network graph: Boost Graph of nodes and edges with UID indexes.
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

#include <knp/core/impexp.h>
#include <knp/core/uid.h>
#include <knp/framework/visualizer/edge.h>
#include <knp/framework/visualizer/node.h>

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include <boost/graph/adjacency_list.hpp>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Boost Graph type definition for network visualization.
 *
 * @details `vecS` as an out-edge list selector allows parallel edges, so several projections between the same pair of
 * populations are kept. `vecS` as a vertex list selector makes a vertex descriptor an index, so the graph has a
 * `vertex_index` property and the index can be used as a node identifier in a DOT file. Vertices are never removed,
 * so descriptors stay valid. `bidirectionalS` gives both `out_edges` and `in_edges`.
 */
using BaseGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, Node, Edge>;


/**
 * @brief Network graph: Boost Graph with UID indexes.
 *
 * @note You can use this to check network structure.
 */
class KNP_DECLSPEC NetworkGraph
{
public:
    using vertex_descriptor = BaseGraph::vertex_descriptor;
    using edge_descriptor = BaseGraph::edge_descriptor;

    /**
     * @brief Construct an empty graph.
     *
     * @param is_dynamic flag showing that the graph is built with message bus subscriptions.
     */
    explicit NetworkGraph(bool is_dynamic = false) : is_dynamic_(is_dynamic) {}

    /**
     * @brief Get graph storage.
     *
     * @return Boost Graph with nodes and edges.
     */
    [[nodiscard]] const BaseGraph &get_graph() const { return graph_; }

    /**
     * @brief Check that the graph was built with message bus subscriptions.
     *
     * @return `true` for a dynamic graph.
     */
    [[nodiscard]] bool is_dynamic() const { return is_dynamic_; }

    /**
     * @brief Add node to the graph and index it by UID.
     *
     * @param node node to add.
     *
     * @return vertex descriptor of the added node.
     */
    vertex_descriptor add_node(const Node &node);

    /**
     * @brief Add edge to the graph and index it by UID.
     *
     * @param from source vertex descriptor.
     * @param to target vertex descriptor.
     * @param edge edge to add.
     *
     * @return edge descriptor of the added edge.
     */
    edge_descriptor add_edge(vertex_descriptor from, vertex_descriptor to, const Edge &edge);

    /**
     * @brief Add invisible node for an edge without source or destination.
     *
     * @return vertex descriptor of the added node.
     *
     * @details A graphviz can't draw an arrow from nowhere. It is necessary to make an invisible vertex.
     */
    vertex_descriptor add_placeholder_node();

    /**
     * @brief Find node by UID.
     *
     * @param uid UID of the node.
     *
     * @return vertex descriptor, or `std::nullopt` if there is no such node.
     */
    [[nodiscard]] std::optional<vertex_descriptor> find_node(const knp::core::UID &uid) const;

    /**
     * @brief Find all edges of a projection.
     *
     * @param uid UID of the projection.
     *
     * @return vector of edge descriptors. One projection can have several senders on the bus and thus several edges.
     */
    [[nodiscard]] std::vector<edge_descriptor> find_edges(const knp::core::UID &uid) const;

private:
    /**
     * @brief Graph storage.
     */
    BaseGraph graph_;

    /**
     * @brief Map UID to vertex descriptor. Invisible nodes are not indexed.
     */
    std::map<knp::core::UID, vertex_descriptor> map_uid_to_nodes_descriptor;

    /**
     * @brief Map UID to edge descriptors. One projection can produce several edges, so the map is a multimap.
     */
    std::multimap<knp::core::UID, edge_descriptor> map_uid_to_edges_descriptor;

    /**
     * @brief A flag showing that the graph was built with message bus subscriptions.
     */
    bool is_dynamic_ = false;

    /**
     * @brief Counter of invisible nodes, used for their names.
     */
    size_t ghost_counter_ = 0;
};

}  // namespace knp::framework::visualizer
