#pragma once
#include <knp/framework/model.h>
#include <knp/framework/network.h>

#include <memory>
#include <string>
#include <vector>

/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief Network description structure used for visualization.
 *
 * @details The structure stores a flat list of population nodes and projection edges, together with their identifiers,
 * names and types. It is constructed from a @ref Network object and then used by the visualizer to build adjacency
 * lists, draw sub‑graphs and compute node positions.
 *
 * @note You can use this to check network structure.
 */
struct KNP_DECLSPEC NetworkGraph
{
public:
    /**
     * @brief Description of a population node.
     *
     * @details Each node corresponds to a population in the original network. The fields store the population size, its
     * unique identifier, a human‑readable name and the neuron type index (used only for drawing legends).
     */
    struct Node
    {
        /**
         * @brief Population size.
         */
        // cppcheck-suppress unusedStructMember
        size_t size_;

        /**
         * @brief Population UID.
         */
        // cppcheck-suppress unusedStructMember
        knp::core::UID uid_;

        /**
         * @brief Population name.
         */
        // cppcheck-suppress unusedStructMember
        std::string name_;

        /**
         * @brief Neuron type.
         */
        // cppcheck-suppress unusedStructMember
        size_t type_;

        /**
         * @brief A flag showing a dynamic or static node.
         *
         * @details It is needed for the graph extension.
         */
        // cppcheck-suppress unusedStructMember
        bool is_static = true;

        /**
         * @brief A flag showing a visible or invisible node.
         *
         * @details It is needed for the graph extension. ( for drawing edges without src or dst )
         */
        // cppcheck-suppress unusedStructMember
        bool is_invisible = false;
    };

    /**
     * @brief Vector of population nodes.
     */
    // cppcheck-suppress unusedStructMember
    std::vector<Node> nodes_;

    /**
     * @brief Description of a projection edge.
     *
     * @details An edge connects a source population (@p index_from_) to a target population (@p index_to_). It stores
     * the projection size, its UID, a readable name and the synapse type index (used for color‑coding in the
     * visualizer).
     */
    struct Edge
    {
        /**
         * @brief Projection size.
         */
        // cppcheck-suppress unusedStructMember
        size_t size_;

        /**
         * @brief Index of the source population.
         */
        // cppcheck-suppress unusedStructMember
        int index_from_;

        /**
         * @brief Index of the target population.
         */
        // cppcheck-suppress unusedStructMember
        int index_to_;

        /**
         * @brief Projection UID.
         */
        // cppcheck-suppress unusedStructMember
        knp::core::UID uid_;

        /**
         * @brief Projection name.
         */
        // cppcheck-suppress unusedStructMember
        std::string name_;

        /**
         * @brief Synapse type.
         */
        // cppcheck-suppress unusedStructMember
        size_t type_;

        /**
         * @brief A flag showing a dynamic or static node.
         *
         * @details It is needed for the graph extension.
         */
        // cppcheck-suppress unusedStructMember
        bool is_static = true;
    };

    /**
     * @brief Vector of projection edges.
     */
    // cppcheck-suppress unusedStructMember
    std::vector<Edge> edges_;

    /**
     * @brief Build network graph from a network.
     *
     * @param network source network for a graph.
     *
     * @details Populations are added as nodes and projections as edges. The constructor extracts UIDs, names and sizes
     * from the network.
     */
    explicit NetworkGraph(const knp::framework::Network& network);
};


/**
 * @brief Print node and edge connections of a network graph (static).
 *
 * @param graph network graph.
 *
 * @details The function writes a textual description of each node (population) and its incoming and outgoing edges to
 * `stdout`. It is primarily useful for debugging the connectivity extraction logic.
 */
KNP_DECLSPEC void print_network_description(const NetworkGraph& graph);

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
KNP_DECLSPEC void visualize_network(const knp::framework::Network& network);

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
KNP_DECLSPEC void visualize_network(const knp::framework::Model& model);

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
KNP_DECLSPEC void visualize_network(
    const knp::framework::Network& network, std::shared_ptr<knp::core::Backend>& backend);

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
KNP_DECLSPEC void visualize_network(const knp::framework::Model& model, std::shared_ptr<knp::core::Backend>& backend);

//TODO:
// KNP_DECLSPEC void visualize_bus(std::shared_ptr<knp::core::Backend>& backend);

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
KNP_DECLSPEC void visualize_bus(const knp::framework::Network& network, std::shared_ptr<knp::core::Backend>& backend);

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
KNP_DECLSPEC void visualize_bus(const knp::framework::Model& model, std::shared_ptr<knp::core::Backend>& backend);

/**
 * @brief Configuration structure for visualization paths.
 */
struct ConfigVisualizePathes
{
    std::string mode = "static";
    std::string name;

    inline static std::string default_path = "visualization_docs";
    std::string dir;
    std::string name_dot_file;
    std::string name_png_file;

    ConfigVisualizePathes() { init(); }

    explicit ConfigVisualizePathes(const std::string& mode_val) : mode(mode_val) { init(); }

private:
    void init()
    {
        name = "graph_" + mode;
        dir = default_path;
        name_dot_file = dir + "/dot_files/" + name + ".dot";
        name_png_file = dir + "/png_files/" + name + ".png";

        try
        {
            std::filesystem::create_directories(dir + "/dot_files");
            std::filesystem::create_directories(dir + "/png_files");
        }
        catch (const std::filesystem::filesystem_error& ex)
        {
            std::cerr << "Failed to create directories: " << ex.what() << std::endl;
        }
    }
};


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
KNP_DECLSPEC void set_saving_path(std::string dir);


}  // namespace knp::framework
