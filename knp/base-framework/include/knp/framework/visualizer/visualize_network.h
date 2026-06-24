/**
 * @file visualize_network.h
 * @brief Functions for graph visualization.
 * @warning Most of the functions are not well-tested or stable yet.
 * @date 26.07.2024
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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
#include <knp/framework/network.h>

#include <string>
#include <vector>

#include <opencv2/core/types.hpp>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief Network description structure used for visualization.
 *
 * @details The structure stores a flat list of population nodes and projection edges, together with their identifiers, 
 * names and types. It is constructed from a @ref Network object and then used by the visualizer to build adjacency lists, 
 * draw sub‑graphs and compute node positions.
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
    };

    /**
     * @brief Vector of population nodes.
     */
    // cppcheck-suppress unusedStructMember
    std::vector<Node> nodes_;

    /**
     * @brief Description of a projection edge.
     *
     * @details An edge connects a source population (@p index_from_) to a target population (@p index_to_). It stores the
     *  projection size, its UID, a readable name and the synapse type index (used for color‑coding in the visualizer).
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
     * @details Populations are added as nodes and projections as edges. The constructor extracts UIDs, names and sizes from
     *  the network.
     */
    explicit NetworkGraph(const knp::framework::Network &network);
};


/**
 * @brief Print node and edge connections of a network graph.
 *
 * @param graph network graph.
 *
 * @details The function writes a textual description of each node (population) and its incoming and outgoing edges to `stdout`.
 * It is primarily useful for debugging the connectivity extraction logic.
 */
KNP_DECLSPEC void print_network_description(const NetworkGraph &graph);


/**
 * @brief Print whole network information.
 * 
 * @param graph network graph.
 * 
 * @note The output format is not intended for end‑users; it is a raw dump useful for developers.
 */
KNP_DECLSPEC void print_modified_network_description(const NetworkGraph &graph);


/**
 * @brief Divide a network graph into independent sub‑graphs.
 *
 * @param graph network graph.
 *
 * @return vector of sub‑graphs, each represented by a list of node indexes.
 *
 * @details The function builds an adjacency list, creates a reverse list for fast inbound look‑ups, and then repeatedly 
 * extracts maximal connected components (ignoring the artificial input node). The resulting sets are sorted for 
 * deterministic ordering.
 */
KNP_DECLSPEC std::vector<std::vector<int>> divide_graph_by_connectivity(const NetworkGraph &graph);


/**
 * @brief Compute positions of nodes in a sub‑graph.
 *
 * @param graph full network graph.
 * @param nodes indexes of the nodes that belong to the sub‑graph.
 * @param screen_size output window size.
 * @param margin border size for the network graph, in pixels.
 * @param num_iterations number of iterations for the force‑directed layout algorithm.
 *
 * @return coordinates of the nodes after layout.
 *
 * @details The function runs the physics‑based layout for @p num_iterations steps and then rescales the resulting positions 
 * to fit inside @p screen_size with the requested @p margin.
 */
KNP_DECLSPEC std::vector<cv::Point2i> position_network(
    const NetworkGraph &graph, const std::vector<int> &nodes, cv::Size screen_size, int margin, int num_iterations);


/**
 * @brief Visualize the iterative positioning of a sub‑graph.
 *
 * @param graph base network graph.
 * @param nodes indexes of the nodes that belong to the sub‑graph.
 * @param screen_size output image size.
 * @param margin size of borders in pixels (default = 50).
 *
 * @details The function opens an OpenCV window and repeatedly:
 *          1. Scales the current graph to the screen
 *          2. Draws the annotated sub‑graph
 *          3. Displays the image
 *          4. Advances the physics simulation by one iteration.
 * 
 * @note Press **Esc** to exit the visualization.         
 */
KNP_DECLSPEC void position_network_test(
    const NetworkGraph &graph, const std::vector<int> &nodes, const cv::Size &screen_size, int margin = 50);
}  // namespace knp::framework
