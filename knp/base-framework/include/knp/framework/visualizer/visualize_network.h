/**
 * @file visualize_network.h
 * @brief Network graph building, description and visualization.
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

#include <knp/core/message_endpoint.h>
#include <knp/core/uid.h>
#include <knp/framework/model.h>
#include <knp/framework/network.h>

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

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
    std::optional<size_t> type_;  // Neuron type.
    std::optional<std::string> comment_;

    /**
     * @brief Node kind as it is written to the node label: "Population", "Modificator", "Input channel" or "Output".
     */
    std::string node_type_ = "Population";

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
    std::optional<size_t> type_;  // Synapse type.
    std::optional<std::string> comment_;

    /**
     * @brief Edge kind as it is written to the edge label: "Projection" or "Spike message".
     */
    std::string edge_type_ = "Projection";

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
    vertex_descriptor add_node(const Node& node)
    {
        auto v = boost::add_vertex(node, graph_);
        if (!node.is_invisible_) map_uid_to_nodes_descriptor[node.uid_] = v;
        return v;
    }

    /**
     * @brief Add edge to the graph and index it by UID.
     *
     * @param from source vertex descriptor.
     * @param to target vertex descriptor.
     * @param edge edge to add.
     *
     * @return edge descriptor of the added edge.
     */
    edge_descriptor add_edge(vertex_descriptor from, vertex_descriptor to, const Edge& edge)
    {
        auto e = boost::add_edge(from, to, edge, graph_).first;
        map_uid_to_edges_descriptor.emplace(edge.uid_, e);
        return e;
    }

    /**
     * @brief Add invisible node for an edge without source or destination.
     *
     * @return vertex descriptor of the added node.
     *
     * @details A graphviz can't draw an arrow from nowhere. It is necessary to make an invisible vertex.
     */
    vertex_descriptor add_placeholder_node()
    {
        Node ghost_node;
        ghost_node.name_ = "ghost_node_" + std::to_string(++ghost_counter_);
        ghost_node.comment_ = "ghost";
        ghost_node.is_invisible_ = true;
        return add_node(ghost_node);
    }

    /**
     * @brief Find node by UID.
     *
     * @param uid UID of the node.
     *
     * @return vertex descriptor, or `std::nullopt` if there is no such node.
     */
    [[nodiscard]] std::optional<vertex_descriptor> find_node(const knp::core::UID& uid) const
    {
        auto it = map_uid_to_nodes_descriptor.find(uid);
        if (it == map_uid_to_nodes_descriptor.end()) return std::nullopt;
        return it->second;
    }

    /**
     * @brief Find all edges of a projection.
     *
     * @param uid UID of the projection.
     *
     * @return vector of edge descriptors. One projection can have several senders on the bus and thus several edges.
     */
    [[nodiscard]] std::vector<edge_descriptor> find_edges(const knp::core::UID& uid) const
    {
        std::vector<edge_descriptor> result;
        auto [first, last] = map_uid_to_edges_descriptor.equal_range(uid);
        for (auto it = first; it != last; ++it) result.push_back(it->second);
        return result;
    }

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
 * @brief Copy subscriptions of a message endpoint.
 *
 * @param endpoint message endpoint of a backend.
 *
 * @return connections that do not depend on the backend lifetime.
 *
 * @details Subscriptions are the wiring registered in the backend, not the message traffic. They are created when a
 * model is loaded and live as long as the backend does, so they can be copied at any moment after the loading.
 */
KNP_DECLSPEC BusConnections get_bus_connections(knp::core::MessageEndpoint& endpoint);


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


/**
 * @brief Build network graph from a network (static structure).
 *
 * @param network source network for a graph.
 *
 * @return network graph.
 *
 * @details Populations are added as nodes and projections as edges. A projection with an unknown presynaptic or
 * postsynaptic UID is skipped, so the graph contains known static connections only.
 * Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network);


/**
 * @brief Build network graph from a model (static structure).
 *
 * @param model source model for a graph.
 *
 * @return network graph.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model);


/**
 * @brief Build network graph from a network and message bus subscriptions (static and dynamic structure).
 *
 * @param network source network for a graph.
 * @param endpoint message endpoint of a backend.
 *
 * @return network graph.
 *
 * @details Connections are taken from the subscriptions only, because a loaded model rebuilds its projections and the
 * presynaptic and postsynaptic UIDs of the network do not match the real connections. The network gives names, sizes
 * and types. A spike message gives a projection source, a synaptic impact message gives a projection destination.
 * Entities that are not described by the network are added as nodes of type "Input channel" or "Output", a population
 * that sends spikes and receives no synaptic impacts is marked as "Modificator". A projection that has no
 * subscriptions at all is skipped.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network, knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build network graph from a model and message bus subscriptions (static and dynamic structure).
 *
 * @param model source model for a graph.
 * @param endpoint message endpoint of a backend.
 *
 * @return network graph.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model, knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build network graph from a network, message bus subscriptions and entities known to the calling code.
 *
 * @param network source network for a graph.
 * @param info endpoint of a backend and entities that cannot be restored from the bus.
 *
 * @return network graph.
 *
 * @details Works as the overload with an endpoint. In addition, an entity from @p info gets its own node kind, and
 * the senders listed for it are drawn as incoming edges. An entity that is not listed is drawn as "External".
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network, const DynamicInfo& info);


/**
 * @brief Build network graph from a model, message bus subscriptions and entities known to the calling code.
 *
 * @param model source model for a graph.
 * @param info endpoint of a backend and entities that cannot be restored from the bus.
 *
 * @return network graph.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model, const DynamicInfo& info);


/**
 * @brief Print node and edge connections of a network graph.
 *
 * @param graph network graph.
 *
 * @details The function writes a textual description of each node and its incoming and outgoing edges to the log.
 * Nodes are grouped by connected components, so independent parts of a network are printed separately.
 */
KNP_DECLSPEC void print_network_description(const NetworkGraph& graph);


/**
 * @brief Write network graph to a stream in DOT format.
 *
 * @param graph network graph.
 * @param out output stream.
 */
KNP_DECLSPEC void write_dot(const NetworkGraph& graph, std::ostream& out);


/**
 * @brief Render network graph to a stream as an image.
 *
 * @param graph network graph.
 * @param out output stream.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @return boolean indicating success.
 *
 * @details No intermediate file is created: the DOT text and the rendered image are kept in memory.
 */
KNP_DECLSPEC bool render_graph(const NetworkGraph& graph, std::ostream& out, const std::string& format = "png");


/**
 * @brief Save DOT and image files of a network graph.
 *
 * @param graph network graph.
 * @param name file name without extension.
 * @param dir directory for saving, it is created if it does not exist.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @return path to the image file, or an empty path if rendering failed. The DOT file is saved in any case.
 *
 * @details Files are saved as `<dir>/dot_files/<name>.dot` and `<dir>/<format>_files/<name>.<format>`.
 */
KNP_DECLSPEC std::filesystem::path visualize(
    const NetworkGraph& graph, const std::string& name = "graph",
    const std::filesystem::path& dir = "visualization_docs", const std::string& format = "png");

}  // namespace knp::framework
