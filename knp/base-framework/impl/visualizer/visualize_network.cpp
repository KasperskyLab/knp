
#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <memory>

extern "C"
{
#include <graphviz/gvc.h>
}
#include <filesystem>

#include "generate_files.h"
#include "graph_helpers.h"


namespace knp::framework
{
/**
 * @brief Set directory for saving visualization files.
 *
 * @param dir directory path for saving.
 *
 * @details Set directory for saving visualization files.
 * Directory "visualization" is being created and it contains two subdirectories for dot files and png files.
 * If the directory does not exist, it will be created.
 * If the first character is /, then the path is absolute. (format: /path/to/absolute/dir)
 * if not, relative to the current directory.  format: path/to/relative/dir)
 */
void set_saving_path(std::string dir)
{
    if (!dir.empty() && dir[0] != '/')
    {
        try
        {
            if (dir.back() == '/') dir.pop_back();

            std::filesystem::path current_dir = std::filesystem::current_path();
            ConfigVisualizePathes::default_path = (current_dir / dir).string();
        }
        catch (const std::filesystem::filesystem_error& e)
        {
            SPDLOG_ERROR("Failed to get current path: {}", e.what());
            ConfigVisualizePathes::default_path = dir;
        }
        catch (...)
        {
            SPDLOG_ERROR("Unknown error getting current path");
            ConfigVisualizePathes::default_path = dir;
        }
    }
    else
    {
        knp::framework::ConfigVisualizePathes::default_path = dir;
    }
}

/**
 * @brief Build network graph from a network.
 * @param network source network for a graph.
 * @details Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
 */
NetworkGraph::NetworkGraph(const knp::framework::Network& network)
{
    // Add populations as nodes.
    for (const auto& pop : network.get_populations())
    {
        size_t pop_size = std::visit([](const auto& p) { return p.size(); }, pop);
        knp::core::UID uid = std::visit([](const auto& p) { return p.get_uid(); }, pop);
        nodes_.push_back(Node{pop_size, uid, get_population_name(pop), pop.index()});
    }

    // Add projections as edges.
    for (const auto& proj : network.get_projections())
    {
        size_t proj_size = std::visit([](const auto& p) { return p.size(); }, proj);
        knp::core::UID uid = std::visit([](const auto& p) { return p.get_uid(); }, proj);
        knp::core::UID uid_from = std::visit([](const auto& p) { return p.get_presynaptic(); }, proj);
        knp::core::UID uid_to = std::visit([](const auto& p) { return p.get_postsynaptic(); }, proj);
        int id_from = -1, id_to = -1;
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            if (uid_from == nodes_[i].uid_) id_from = i;
            if (uid_to == nodes_[i].uid_) id_to = i;
        }
        edges_.push_back(Edge{proj_size, id_from, id_to, uid, get_population_name(proj), proj.index()});
    }
}
/**
 * @brief Print node and edge connections of a network graph.
 *
 * @param graph network graph.
 *
 * @details The function writes a textual description of each node (population) and its incoming and outgoing edges to
 * `stdout`. It is primarily useful for debugging the connectivity extraction logic. (needed to static network)
 */
void print_network_description(const NetworkGraph& graph)
{
    // out to console
    // for (size_t i = 0; i < graph.nodes_.size(); ++i)
    // {
    //     std::cout << "Population #" << i << " of size " << graph.nodes_[i].size_ << ": receive from";

    // std::vector<int> edges_to_i;
    // std::vector<int> edges_from_i;

    // for (const auto& edge : graph.edges_)
    // {
    //     if (edge.index_to_ == static_cast<int>(i)) edges_to_i.push_back(edge.index_from_);
    //     if (edge.index_from_ == static_cast<int>(i)) edges_from_i.push_back(edge.index_to_);
    // }
    // for (auto edge : edges_to_i) std::cout << " #" << edge;
    // std::cout << "; send to";
    // for (auto edge : edges_from_i) std::cout << " #" << edge;
    // std::cout << std::endl;
    //}
    //out to log info
    for (size_t i = 0; i < graph.nodes_.size(); ++i)
    {
        std::string log_msg = fmt::format("Population #{} of size {},: receive from", i, graph.nodes_[i].size_);


        std::vector<int> edges_to_i;
        std::vector<int> edges_from_i;

        for (const auto& edge : graph.edges_)
        {
            if (edge.index_to_ == static_cast<int>(i)) edges_to_i.push_back(edge.index_from_);
            if (edge.index_from_ == static_cast<int>(i)) edges_from_i.push_back(edge.index_to_);
        }
        for (auto edge : edges_to_i) log_msg += fmt::format(" #{}", edge);
        log_msg += fmt::format("; send to");
        for (auto edge : edges_from_i) log_msg += fmt::format(" #{}", edge);
        SPDLOG_INFO(log_msg);
    }
}

/**
 * @brief Visualize static network.
 *
 * @param network source network for visualization.
 *
 * @details Visualize network.
 * The model is needed to get the names of static nodes.
 * By default, it saves dot/png files of graph to the current directory (the visualization directory is being created).
 * For the change save directory, use set_saving_path(directory).
 */
void visualize_network(const knp::framework::Network& network)
{
    knp::framework::NetworkGraph graph(network);
    knp::framework::ConfigVisualizePathes file_info{};

    create_dot_file_for_static_network(file_info.name_dot_file, graph);
    create_png_file(file_info.name_dot_file, file_info.name_png_file);
    print_network_description(graph);  // description of static network
}
/**
 * @brief Visualize static network by model.
 *
 * @param model source model for visualization.
 *
 * @details Visualize network.
 * The model is needed to get the names of static nodes.
 * By default, it saves dot/png files of graph to the current directory (the visualization directory is being created).
 * For the change save directory, use set_saving_path(directory).
 */
void visualize_network(const knp::framework::Model& model)
{
    visualize_network(model.get_network());
}

/**
 * @brief Visualize dynamic network using backend.
 *
 * @param network source network for visualization.
 * @param backend visualization backend.
 *
 * @details Visualize network.
 * The network is needed to get the names of static nodes.
 * The backend is needed to get all graphs connections from the bus.
 * By default, it saves dot/png files of graph to the current directory (the visualization directory is being created).
 * For the change save directory, use set_saving_path(directory).
 */
void visualize_network(const knp::framework::Network& network, std::shared_ptr<knp::core::Backend>& backend)
{
    knp::framework::NetworkGraph graph(network);
    knp::framework::ConfigVisualizePathes file_info{"dynamic"};

    create_dot_file_for_dynamic_network(file_info.name_dot_file, graph, backend);
    create_png_file(file_info.name_dot_file, file_info.name_png_file);
}
/**
 * @brief Visualize dynamic network using backend.
 *
 * @param model source model for visualization.
 * @param backend visualization backend.
 *
 * @details Visualize network.
 * The model is needed to get the names of static nodes.
 * The backend is needed to get all graphs connections from the bus.
 * By default, it saves dot/png files of graph to the current directory (the visualization directory is being created).
 * For the change save directory, use set_saving_path(directory).
 */
void visualize_network(const knp::framework::Model& model, std::shared_ptr<knp::core::Backend>& backend)
{
    visualize_network(model.get_network(), backend);
}

/**
 * @brief Visualize the bus messages.
 *
 * @param network source network for bus visualization.
 * @param backend visualization backend.
 *
 * @details Visualize the bus messages. The network is needed to get the names of static nodes.
 * By default, it saves dot/png files of graph to the current directory (the visualization directory is being created).
 * For the change save directory, use set_saving_path(directory).
 */
void visualize_bus(const knp::framework::Network& network, std::shared_ptr<knp::core::Backend>& backend)
{
    knp::framework::NetworkGraph graph(network);
    knp::framework::ConfigVisualizePathes file_info{"bus"};

    create_dot_file_for_bus(file_info.name_dot_file, graph, backend);
    create_png_file(file_info.name_dot_file, file_info.name_png_file);
}
/**
 * @brief Visualize the bus messages.
 *
 * @param model source model for bus visualization.
 * @param backend visualization backend.
 *
 * @details Visualize the bus messages. The model is needed to get the names of static nodes.
 * By default, it saves dot/png files of graph to the current directory (the visualization directory is being created).
 * For the change save directory, use set_saving_path(directory).
 */
void visualize_bus(const knp::framework::Model& model, std::shared_ptr<knp::core::Backend>& backend)
{
    visualize_bus(model.get_network(), backend);
}

}  // namespace knp::framework
