#include "generate_files.h"

#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dot_helpers.h"
#include "graph_helpers.h"

extern "C"
{
#include <graphviz/gvc.h>
}

/**
 * @brief Create DOT file for static network
 *
 * @param file_name output file name
 * @param graph network graph
 */
void create_dot_file_for_static_network(std::string& file_name, knp::framework::NetworkGraph& graph)
{
    std::ofstream out(file_name);

    if (!out.is_open())
    {
        SPDLOG_ERROR("Cannot open file {} for writing", file_name);
        return;
    }

    write_header_and_styles(out);
    for (size_t i = 0; i < graph.nodes_.size(); ++i)
    {
        auto node = graph.nodes_[i];
        //write_node_to_dot(out, i, node.name_, node.size_);
        out << "    " << i << " [label=\"population:\\n"
            << node.name_ << "\\nsize: " << node.size_ << "\", color=\""
            << "\"];\n";
    }
    for (const auto& edge : graph.edges_)
    {
        if (edge.index_from_ == -1)
        {
            // write_edge_to_dot(out, "00000000", edge.index_to_, edge.name_);
            // out << "    " << "00000000" << " -> " <<  edge.index_to_ << " [label=\"projection :\\n" << edge.name_ <<
            // "\"];\n";
            continue;
        }
        // write_edge_to_dot(out, edge.index_from_, edge.index_to_, edge.name_);
        out << "    " << edge.index_from_ << " -> " << edge.index_to_ << " [label=\"projection :\\n"
            << edge.name_ << "\"];\n";
    }
    out << "}\n";
    if (out.good())
    {
        SPDLOG_INFO("The DOT file is saved in: {}", std::filesystem::absolute(file_name).string());
    }
    else
    {
        SPDLOG_ERROR("Failed to write to file {}", file_name);
    }
}

/**
 * @brief Create DOT file for bus messages
 *
 * @param file_name output DOT file name
 * @param graph network graph
 * @param backend shared pointer to backend
 */
void create_dot_file_for_bus(
    const std::string& file_name, const knp::framework::NetworkGraph& graph,
    std::shared_ptr<knp::core::Backend>& backend)
{
    auto [spike_messages, synaptic_messages] = get_bus(backend);

    std::ofstream out(file_name);

    write_bus_messeges_to_dot(spike_messages, synaptic_messages, graph, out);  // spike/synaptic impact messeges

    SPDLOG_INFO("The DOT file is saved in: {}", std::filesystem::absolute(file_name).string());
}

/**
 * @brief Create DOT file for dynamic network
 *
 * @param file_name output file name
 * @param graph network graph
 * @param backend shared pointer to backend
 */
void create_dot_file_for_dynamic_network(
    const std::string& file_name, const knp::framework::NetworkGraph& graph,
    std::shared_ptr<knp::core::Backend>& backend)
{
    const auto subs = backend->get_message_endpoint().get_endpoint_subscriptions();

    constexpr size_t SPIKE_IDX = 0;            // must be 0
    constexpr size_t SYNAPTIC_IMPACT_IDX = 1;  // must be 1

    std::map<knp::core::UID, knp::core::UID> node_src;
    std::map<knp::core::UID, knp::core::UID> node_dst;

    std::set<knp::core::UID> unique_senders_spike;
    std::set<knp::core::UID> unique_receivers_impact;

    std::set<knp::core::UID> unique_nodes;
    std::set<knp::core::UID> unique_edges;


    std::vector<std::pair<knp::core::UID, knp::core::UID>> spike_messages;     // [sender, receiver]
    std::vector<std::pair<knp::core::UID, knp::core::UID>> synaptic_messages;  // [sender, receiver]


    // take from bus nessosary maps
    for (const auto& [key, sub_variant] : subs)
    {
        const auto& [type_idx, receiver_uid] = key;

        std::visit(
            [&](const auto& sub)
            {
                const auto& senders = sub.get_senders();
                if (senders.empty()) return;

                if (type_idx == SPIKE_IDX)
                {  // receiver: Projection,      sender: *Population     (* - or modificator) // 0 -->
                    for (const auto& sender : senders)
                    {
                        spike_messages.push_back({sender, receiver_uid});
                        unique_senders_spike.insert(sender);
                        node_src[receiver_uid] = sender;
                        // edge_from[sender] = receiver_uid;

                        unique_nodes.insert(sender);
                        unique_edges.insert(receiver_uid);
                    }
                }
                else if (type_idx == SYNAPTIC_IMPACT_IDX)
                {  // receiver: *Population,      sender: Projection    (* - or modificator)  //  --> 0
                    for (const auto& sender : senders)
                    {
                        synaptic_messages.push_back({sender, receiver_uid});
                        unique_receivers_impact.insert(receiver_uid);
                        node_dst[sender] = receiver_uid;
                        // edge_to[receiver_uid] = sender;
                        unique_nodes.insert(receiver_uid);
                        unique_edges.insert(sender);
                    }
                }
            },
            sub_variant);
    }


    std::set<knp::core::UID> modificators;


    // modificator exist in senders SPIKE
    // modificator doesn't exist in receivers SYNAPTIC_IMPACT
    for (const auto& uid : unique_senders_spike)
    {
        if (unique_receivers_impact.find(uid) == unique_receivers_impact.end())
        {
            modificators.insert(uid);
            SPDLOG_INFO("Modificator: {}", std::string(uid));
        }
    }

    ////////////////////////////////////////

    std::ofstream out(file_name);
    std::string dynamic_color = "yellow";
    write_projections_and_populations_to_dot(
        unique_nodes, unique_edges, modificators, out, dynamic_color, graph, node_src,
        node_dst);  // projections/populations from bus

    SPDLOG_INFO("The DOT file is saved in: {}", std::filesystem::absolute(file_name).string());
}

/////////////////  dot to png /////////////////////
/**
 * @brief Convert DOT file to PNG image
 *
 * @param path_to_dot_file path to input DOT file
 * @param path_to_png_file path to output PNG file
 *
 * @return boolean indicating success
 */
bool convert_dot_to_png(const std::string& path_to_dot_file, const std::string& path_to_png_file)
{
    // 1. Initialize the Graphviz context
    auto gvc = std::unique_ptr<GVC_t, decltype(&gvFreeContext)>(gvContext(), gvFreeContext);
    if (!gvc) return false;

    // 2. Open and parse the DOT file
    // auto fp = std::unique_ptr<FILE, decltype(&fclose)>(fopen(path_to_dot_file.c_str(), "r"), fclose);
    struct FileDeleter
    {
        void operator()(FILE* fp) const
        {
            if (fp)
            {
                fclose(fp);
            }
        }
    };
    std::unique_ptr<FILE, FileDeleter> fp(fopen(path_to_dot_file.c_str(), "r"));

    if (!fp)
    {
        SPDLOG_ERROR("Error: couldn't open the DOT file: {}", std::filesystem::absolute(path_to_dot_file).string());
        return false;
    }

    // 3. Read the DOT file
    auto g = std::unique_ptr<Agraph_t, decltype(&agclose)>(agread(fp.get(), nullptr), agclose);
    if (!g)
    {
        SPDLOG_ERROR("Error: couldn't read DOT file: {}", std::filesystem::absolute(path_to_dot_file).string());
        return false;
    }

    // 4. Compute the layout using the "dot" engine
    // gvLayoutJobs(gvc.get(), g.get());
    int layout_res = gvLayoutJobs(gvc.get(), g.get());
    if (layout_res != 0)
    {
        SPDLOG_ERROR("Layout computation failed");
        return false;
    }
    // gvLayout(gvc.get(), g.get(), "dot");

    // 5. Render the layout into a PNG file
    int result = gvRenderFilename(gvc.get(), g.get(), "png", path_to_png_file.c_str());


    // 6. Check the result and print the path to the PNG file
    if (result == 0)
    {
        SPDLOG_INFO("The PNG file is saved in: {}", std::filesystem::absolute(path_to_png_file).string());
        return true;
    }
    else
    {
        SPDLOG_ERROR("PNG file rendering error");
        return false;
    }
}

/**
 * @brief Create PNG file from DOT file
 *
 * @param dot_file path to DOT file
 * @param png_file path to PNG file
 */
void create_png_file(const std::string& dot_file, const std::string& png_file)
{
    // system("dot -Tpng ./store_log_file3/graph.dot -o ./store_log_file3/graph.png");
    if (!convert_dot_to_png(dot_file, png_file)) SPDLOG_ERROR("Dot file conversion failed.");
}
