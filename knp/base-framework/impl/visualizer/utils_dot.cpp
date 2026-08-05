#include "utils_dot.h"

#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils_graph.h"


/**
 * @brief Write header and styles to DOT file
 *
 * @param out output file stream
 *
 * @details commentsed - for a connected graph
 */
void write_header_and_styles(std::ofstream& out)
{
    // digraph - directed
    out << "digraph G {\n";

    // for unconnected graphs this is needed: (+ change -> to --)
    out << "    layout=neato;\n";          // Change engine to neato (physical force model)
    out << "    pack=true;\n";             // Enable compact packing of independent components
    out << "    packmode=\"graph\";\n\n";  // Pack each subgroup as a separate mini-graph
                                           //

    // Global design settings for the graph
    out << "    // Canvas settings\n";
    out << "    bgcolor=\"#FAFAFA\";\n";  // Light gray soft background page
    // out << "    rankdir=LR;\n";                 // Graph goes left to right
    out << "    splines=true;\n";  // Beautiful smooth curves for arrows
    // out << "    nodesep=0.5;\n";                // Distance between nodes
    // out << "    ranksep=0.6;\n\n";              // Distance between levels (layers) of the graph
    out << "    overlap=false;\n";
    out << "    sep=\"+30\";\n";

    out << "    // Default node styling\n";
    out << "    node [\n";
    out << "        fontname=\"Helvetica,Arial,sans-serif\",\n";  // Modern font
    out << "        fontsize=11,\n";
    out << "        shape=box,\n";                 // Rectangular shape
    out << "        style=\"filled,rounded\",\n";  // Fill with color + rounded corners
    out << "        penwidth=0,\n";                // Remove rigid black border around node
    out << "        margin=\"0.2,0.1\"\n";         // Internal padding for text
    out << "    ];\n\n";

    out << "    // Default edge (arrow) styling\n";
    out << "    edge [\n";
    out << "        fontname=\"Helvetica,Arial,sans-serif\",\n";
    out << "        fontsize=9,\n";
    out << "        fontcolor=\"#555555\",\n";  // Soft gray color for text above arrows
    out << "        color=\"#B0BEC5\",\n";      // Beautiful steel color for the arrow itself
    out << "        penwidth=1.5,\n";           // Make line slightly thicker than default
    out << "        arrowsize=0.8\n";           // Slightly reduce arrowhead size
    out << "    ];\n\n";
}

/**
 * @brief Write edge without source to DOT file
 *
 * @param out output file stream
 * @param dst destination node UID
 * @param i counter for ghost node
 * @param edge_uid edge UID
 * @param size edge size
 * @param color edge color
 *
 * @details A graphviz can't draw an arrow from nowhere. It is necessary to make an invisible vertex.
 */
void write_edge_without_src(
    std::ofstream& out, knp::core::UID const& dst, int const& i, knp::core::UID const& edge_uid,
    std::string const& size, std::string const& color)
{
    auto num = std::to_string(i);
    auto short_name = std::string(edge_uid).substr(0, 8);


    out << "    "
        << "ghost_src_" << num << " [shape=point, style=invis"
        << "];\n";


    out << "    \""
        << "ghost_src_" << num << "\" -> " << std::string(dst) << "[style=\"dotted\", "
        << "label=\"projection :\\n"
        << short_name << "\\nsize:" << size << "\", color=\"" << color << "\"];\n";
}

/**
 * @brief Write edge without destination to DOT file
 *
 * @param out output file stream
 * @param src source node UID
 * @param i counter for ghost node
 * @param edge_uid edge UID
 * @param size edge size
 * @param color edge color
 *
 * @details A graphviz can't draw an arrow to nowhere. It is necessary to make an invisible vertex.
 */
void write_edge_without_dst(
    std::ofstream& out, knp::core::UID const& src, int const& i, knp::core::UID const& edge_uid,
    std::string const& size, std::string const& color)
{
    auto num = std::to_string(i);
    auto short_name = std::string(edge_uid).substr(0, 8);

    out << "    "
        << "ghost_dst_" << num << " [shape=point, style=invis"
        << "];\n";

    out << "    \"" << std::string(src) << "\" -> "
        << "ghost_dst_" << num << "[style=\"dotted\", "
        << "label=\"projection :\\n"
        << short_name << "\\nsize:" << size << "\", color=\"" << color << "\"];\n";
}

/**
 * @brief Write bus messages (spike messages and synaptic impact messages) to DOT file
 *
 * @param spike_messages vector of spike message pairs
 * @param synaptic_messages vector of synaptic message pairs
 * @param graph network graph
 * @param out output file stream for DOT file
 */
void write_bus_messeges_to_dot(
    std::vector<std::pair<knp::core::UID, knp::core::UID>> const& spike_messages,
    std::vector<std::pair<knp::core::UID, knp::core::UID>> const& synaptic_messages,
    knp::framework::NetworkGraph const& graph, std::ofstream& out)
{
    write_header_and_styles(out);

    for (auto spike_message : spike_messages)
    {
        auto spike_sender = spike_message.first;
        auto spike_receiver = spike_message.second;
        auto sender_name = std::string(spike_sender).substr(0, 8);
        auto receiver_name = std::string(spike_receiver).substr(0, 8);
        std::string sender_size = "";
        std::string receiver_size = "";

        auto node_it = std::find_if(
            graph.nodes_.begin(), graph.nodes_.end(),
            [&spike_sender](auto const& node) { return node.uid_ == spike_sender; });
        if (node_it != graph.nodes_.end())
        {
            sender_name = node_it->name_;
            sender_size = std::to_string(node_it->size_);
        }

        auto edge_it = std::find_if(
            graph.edges_.begin(), graph.edges_.end(),
            [&spike_receiver](const auto& edge) { return edge.uid_ == spike_receiver; });
        if (edge_it != graph.edges_.end())
        {
            receiver_name = edge_it->name_;
            receiver_size = std::to_string(edge_it->size_);
        }

        auto index_from = std::string(spike_sender);
        auto index_to = std::string(spike_receiver);

        // write nodes
        out << "    \"" << index_from << "\" [label=\"" << sender_name << "\"];"
            << "# " << sender_size << "\n";
        out << "    \"" << index_to << "\" [label=\"" << receiver_name << "\"];"
            << "# " << receiver_size << "\n";

        // write edge
        out << "    \"" << index_from << "\" -> \"" << index_to << "\" [label=\"Spike Message"
            << "\"];\n";
    }

    for (auto synaptic_message : synaptic_messages)
    {
        auto synaptic_sender = synaptic_message.first;
        auto synaptic_receiver = synaptic_message.second;
        auto sender_name = std::string(synaptic_sender).substr(0, 8);
        auto receiver_name = std::string(synaptic_receiver).substr(0, 8);
        std::string sender_size = "";
        std::string receiver_size = "";

        auto edge_it = std::find_if(
            graph.edges_.begin(), graph.edges_.end(),
            [&synaptic_sender](const auto& edge) { return edge.uid_ == synaptic_sender; });
        if (edge_it != graph.edges_.end())
        {
            sender_name = edge_it->name_;
            sender_size = std::to_string(edge_it->size_);
        }

        auto node_it = std::find_if(
            graph.nodes_.begin(), graph.nodes_.end(),
            [&synaptic_receiver](const auto& node) { return node.uid_ == synaptic_receiver; });
        if (node_it != graph.nodes_.end())
        {
            receiver_name = node_it->name_;
            receiver_size = std::to_string(node_it->size_);
        }

        auto index_from = std::string(synaptic_sender);
        auto index_to = std::string(synaptic_receiver);

        // write nodes
        out << "    \"" << index_from << "\" [label=\"" << sender_name << "\"];"
            << "# " << sender_size << "\n";
        out << "    \"" << index_to << "\" [label=\"" << receiver_name << "\"];"
            << "# " << receiver_size << "\n";

        // write edge
        out << "    \"" << index_from << "\" -> \"" << index_to << "\" [label=\"Synaptic Impact Message"
            << "\"];\n";
    }

    out << "}\n";
}

/**
 * @brief Write projections and populations to DOT file
 *
 * @param unique_nodes set of unique node UIDs
 * @param unique_edges set of unique edge UIDs
 * @param modificators set of modificator UIDs
 * @param out output file stream
 * @param dynamic_color color for dynamic elements
 * @param graph network graph
 * @param node_src map of node sources
 * @param node_dst map of node destinations
 */
void write_projections_and_populations_to_dot(
    std::set<knp::core::UID> const& unique_nodes, std::set<knp::core::UID> const& unique_edges,
    std::set<knp::core::UID>& modificators, std::ofstream& out, std::string& dynamic_color,
    knp::framework::NetworkGraph const& graph, std::map<knp::core::UID, knp::core::UID>& node_src,
    std::map<knp::core::UID, knp::core::UID>& node_dst)
{
    write_header_and_styles(out);

    if (unique_nodes.empty() && unique_edges.empty())
    {
        SPDLOG_WARN("No nodes or edges to visualize");
        out << "}\n";
        return;
    }

    /////// write nodes to dot //////////
    for (const auto& unique_node : unique_nodes)
    {
        // check if it is modificator
        if (modificators.find(unique_node) != modificators.end())
        {
            write_node_to_dot(
                out, std::string(unique_node), std::string(unique_node).substr(0, 8), "None", dynamic_color,
                "Modificator");
            continue;
        }
        auto graph_node = get_graph_node_by_uid(unique_node, graph.nodes_);
        auto node_name = get_node_name(unique_node, graph.nodes_);
        auto graph_node_size = graph_node.size_;
        write_node_to_dot(out, std::string(unique_node), node_name, graph_node_size);
    }

    /////// write edges to dot //////////
    [[maybe_unused]] int num_invisible_src_nodes = 1;
    [[maybe_unused]] int num_invisible_dst_nodes = 1;
    for (const auto& unique_edge : unique_edges)
    {
        auto edge_size = get_proj_size(graph.edges_, unique_edge);
        auto it1 = node_src.find(unique_edge);
        if (it1 == node_src.end())
        {
            SPDLOG_INFO("There is't node_src for this edge: {}", std::string(unique_edge));
            knp::core::UID dst;
            try
            {
                dst = node_dst.at(unique_edge);
            }
            catch (const std::out_of_range& e)
            {
                SPDLOG_ERROR("ERROR: There is an edge without a source and a destination! Details: {}", e.what());
            }
            //write node to dot (ghost_i -> dst)
            write_edge_without_src(out, dst, num_invisible_dst_nodes, unique_edge, edge_size);
            num_invisible_src_nodes++;
            continue;
        }
        auto src = node_src.at(unique_edge);

        auto it2 = node_dst.find(unique_edge);
        if (it2 == node_dst.end())
        {
            SPDLOG_INFO("There is't node_dst for this edge: {}", std::string(unique_edge));
            //write node to dot (src -> ghost_j)
            write_edge_without_dst(out, src, num_invisible_dst_nodes, unique_edge, edge_size);
            num_invisible_dst_nodes++;
            continue;
        }
        auto dst = node_dst.at(unique_edge);

        // check if the source is a modifier (change color of edge)
        if (modificators.find(src) != modificators.end())
        {
            write_edge_to_dot(
                out, std::string(src), std::string(dst), std::string(unique_edge).substr(0, 8), edge_size,
                dynamic_color);
            continue;
        }
        write_edge_to_dot(out, std::string(src), std::string(dst), std::string(unique_edge).substr(0, 8), edge_size);
    }

    out << "}\n";
}
