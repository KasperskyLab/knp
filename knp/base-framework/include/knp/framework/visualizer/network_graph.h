/**
 * @file network_graph.h
 * @brief Network graph types: nodes, edges, graph storage and data for a dynamic graph.
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

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Node kind of a population described by the network.
 */
constexpr char population_node_type[]{"Population"};


/**
 * @brief Edge kind of a projection.
 */
constexpr char projection_edge_type[]{"Projection"};


/**
 * @brief Network node representation.
 *
 * @details Stores population information including size, UID, name and type.
 * The structure is stored as a vertex property in Boost Graph.
 */
struct Node
{
    knp::core::UID uid_;
    std::string name_;
    std::optional<size_t> size_;
    std::optional<size_t> type_;  // Index of the neuron type in `knp::core::AllPopulationsVariant`.
    std::optional<std::string> comment_;

    /**
     * @brief Node kind as it is written to the node label.
     *
     * @details `population_node_type` for a population of the network. A node that is not described by the network
     * gets the kind given by the calling code, or "External" if the calling code does not describe it.
     */
    std::string node_type_ = population_node_type;

    /**
     * @brief A flag showing a dynamic or static node.
     *
     * @details `true` if the node is described by the network, `false` if it is known from the message bus only.
     */
    bool is_static_ = true;

    /**
     * @brief A flag showing a visible or invisible node.
     *
     * @details It is needed for drawing edges without source or destination.
     */
    bool is_invisible_ = false;

    Node() = default;

    Node(
        knp::core::UID uid, std::string name, std::optional<size_t> size, std::optional<size_t> type,
        std::optional<std::string> comment)
        : uid_(uid), name_(std::move(name)), size_(size), type_(type), comment_(std::move(comment))
    {
    }
};


/**
 * @brief Network edge representation.
 *
 * @details Stores projection information including size, connection UIDs, UID, name and type.
 * The structure is stored as an edge property in Boost Graph.
 */
struct Edge
{
    knp::core::UID uid_;
    std::string name_;
    std::optional<knp::core::UID> index_from_;  // Source UID, `std::nullopt` for an invisible source.
    std::optional<knp::core::UID> index_to_;    // Target UID, `std::nullopt` for an invisible target.
    std::optional<size_t> size_;
    std::optional<size_t> type_;  // Index of the synapse type in `knp::core::AllProjectionsVariant`.
    std::optional<std::string> comment_;

    /**
     * @brief Edge kind as it is written to the edge label: `projection_edge_type` or "Spike Message".
     */
    std::string edge_type_ = projection_edge_type;

    /**
     * @brief A flag showing a dynamic or static edge.
     *
     * @details `true` if the edge is described by the network.
     */
    bool is_static_ = false;

    /**
     * @brief A flag showing that the edge is confirmed by message bus subscriptions.
     */
    bool is_dynamic_ = false;

    Edge() = default;

    Edge(
        knp::core::UID uid, std::string name, std::optional<knp::core::UID> index_from,
        std::optional<knp::core::UID> index_to, std::optional<size_t> size, std::optional<size_t> type,
        std::optional<std::string> comment)
        : uid_(uid),
          name_(std::move(name)),
          index_from_(index_from),
          index_to_(index_to),
          size_(size),
          type_(type),
          comment_(std::move(comment))
    {
    }
};


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
     * @brief Add node to the graph and index it by UID.
     *
     * @param node node to add.
     *
     * @return vertex descriptor of the added node.
     */
    vertex_descriptor add_node(const Node& node);

    /**
     * @brief Add edge to the graph and index it by UID.
     *
     * @param from source vertex descriptor.
     * @param to target vertex descriptor.
     * @param edge edge to add.
     *
     * @return edge descriptor of the added edge.
     */
    edge_descriptor add_edge(vertex_descriptor from, vertex_descriptor to, const Edge& edge);

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
    [[nodiscard]] std::optional<vertex_descriptor> find_node(const knp::core::UID& uid) const;

    /**
     * @brief Find all edges of a projection.
     *
     * @param uid UID of the projection.
     *
     * @return vector of edge descriptors. One projection can have several senders on the bus and thus several edges.
     */
    [[nodiscard]] std::vector<edge_descriptor> find_edges(const knp::core::UID& uid) const;

private:
    size_t ghost_counter_ = 0;
};


/**
 * @brief Connections taken from message endpoint subscriptions.
 *
 * @details Plain copies of the subscriptions: [sender, receiver] pairs. The structure keeps no reference to a
 * backend, so it stays valid after the backend is destroyed and can be passed anywhere.
 */
struct BusConnections
{
    /// Spike messages: the sender is a population, a handler or an input channel, the receiver is a projection.
    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<knp::core::UID, knp::core::UID>> spike_messages_;
    /// Synaptic impact messages: the sender is a projection, the receiver is a population.
    // cppcheck-suppress unusedStructMember
    std::vector<std::pair<knp::core::UID, knp::core::UID>> synaptic_messages_;
};


/**
 * @brief Entity that is not described by the network: a message handler or a channel.
 *
 * @details Some entities cannot be restored from the message bus of a backend. A message handler subscribes to its
 * senders through its own message endpoint, and an output channel does the same, so these subscriptions are not
 * registered in the backend. Such entities are described by the calling code.
 */
struct ExternalEntity
{
    /// Entity UID.
    knp::core::UID uid_;
    /// Node kind for the label, for example "WTA", "Input channel" or "Output channel".
    std::string name_ = "External";
    /// Entities the given one listens to. Leave empty if the subscriptions are visible on the bus.
    // cppcheck-suppress unusedStructMember
    std::vector<knp::core::UID> senders_;
};


/**
 * @brief Data for a dynamic graph.
 */
struct DynamicInfo
{
    /// Connections taken from the backend message bus.
    // cppcheck-suppress unusedStructMember
    BusConnections bus_;
    /// Handlers and channels. An entity that is not listed here is drawn as "External".
    // cppcheck-suppress unusedStructMember
    std::vector<ExternalEntity> entities_;
};

}  // namespace knp::framework::visualizer
