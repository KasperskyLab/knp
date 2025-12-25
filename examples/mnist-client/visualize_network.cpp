/**
 * @file visualize_network.cpp
 * @brief Functions for subgraph visualization.
 * @kaspersky_support A. Vartenkov
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

#include "visualize_network.h"

#include <knp/framework/network.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <unordered_set>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "graph_physics.h"

/**
 * @brief Adjacency list is a representation of a graph where there is a list of adjacent nodes for each node.
 * @see https://en.wikipedia.org/wiki/Adjacency_list.
 */
using AdjacencyList = std::vector<std::vector<size_t>>;


/**
 * @brief Parameters for network drawing.
 */
struct DrawingParameters
{
    /**
     * @brief Nodes will be drawn using this color.
     */
    const cv::Scalar node_color{0, 0, 0};
    /**
     * @brief Background color.
     */
    const cv::Scalar back_color{255, 255, 255};
    /**
     * @brief Color for graph edges.
     */
    const cv::Scalar edge_color{0, 0, 255};
    /**
     * @brief Size of a circular arrow pointing from a node to the same node.
     */
    const int self_arrow_radius = 30;
    /**
     * @brief Node circle radius.
     */
    const int node_radius = 10;
    /**
     * @brief Length of an arrow showing the projection target.
     */
    const int arrow_len = 20;
    /**
     * @brief An arrow head is a triangle. Arrow width is back side length divided by `2 * arrow_len` (`1.0` for a right
     * angle).
     */
    const double arrow_width = 0.3;
    /**
     * @brief Minimal number of pixels between image edge and text.
     */
    const int text_margin = 5;
};


/**
 * @brief Get entity name for any network object. If there is no name it's constructed from UID.
 */
template <class Entity>
std::string get_name(const Entity &pop)
{
    const size_t uid_part_size = 8;
    knp::core::TagMap tags = std::visit([](const auto &p) { return p.get_tags(); }, pop);
    knp::core::UID uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
    std::string name;
    auto tag = tags.get_tag("name");
    if (tag.has_value())
    {
        try
        {
            name = std::any_cast<std::string>(tag);
        }
        catch (std::bad_any_cast &exc)
        {
            SPDLOG_WARN("Wrong name tag type.");
        }
    }
    if (name.empty()) name = std::string{uid}.substr(0, uid_part_size);
    return name;
}


/**
 * @brief Build network graph from a network.
 * @param network source network for a graph.
 * @details Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
 */
NetworkGraph::NetworkGraph(const knp::framework::Network &network)
{
    // Add populations as nodes.
    for (const auto &pop : network.get_populations())
    {
        size_t pop_size = std::visit([](const auto &p) { return p.size(); }, pop);
        knp::core::UID uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
        nodes_.push_back(Node{pop_size, uid, get_name(pop), pop.index()});
    }

    // Add projections as edges.
    for (const auto &proj : network.get_projections())
    {
        size_t proj_size = std::visit([](const auto &p) { return p.size(); }, proj);
        knp::core::UID uid = std::visit([](const auto &p) { return p.get_uid(); }, proj);
        knp::core::UID uid_from = std::visit([](const auto &p) { return p.get_presynaptic(); }, proj);
        knp::core::UID uid_to = std::visit([](const auto &p) { return p.get_postsynaptic(); }, proj);
        int id_from = -1, id_to = -1;
        for (size_t i = 0; i < nodes_.size(); ++i)
        {
            if (uid_from == nodes_[i].uid_) id_from = i;
            if (uid_to == nodes_[i].uid_) id_to = i;
        }
        edges_.push_back(Edge{proj_size, id_from, id_to, uid, get_name(proj), proj.index()});
    }
}


/**
 * @brief Convert network graph into adjacency list form.
 * @param graph network graph built from a `Network` object.
 * @return graph as an adjacency list.
 * @see https://en.wikipedia.org/wiki/Adjacency_list.
 * @note There is no input node in `NetworkGraph`, but an adjacency list must have one as the last node.
 */
AdjacencyList build_adjacency_list(const NetworkGraph &graph)
{
    AdjacencyList adj_list;
    adj_list.resize(graph.nodes_.size() + 1);
    for (const auto &edge : graph.edges_)
    {
        int index = edge.index_from_;
        if (index < 0) index = static_cast<int>(graph.nodes_.size());
        adj_list[index].push_back(edge.index_to_);
    }
    return adj_list;
}


/**
 * @brief Draw a line with an arrow that is much nicer than basic OpenCV arrow.
 * @param img image to modify.
 * @param pt_from point from which the arrow starts.
 * @param pt_to point to which the arrow points.
 * @param len arrow length in pixels.
 * @param width arrow width as a fraction of arrow length.
 * @param margin arrow shift from the end of the line to the beginning by "margin" pixels.
 * @param color line and arrow color.
 */
void draw_simple_arrow_line(
    cv::Mat &img, const cv::Point2d &pt_from, const cv::Point2d &pt_to, int len, double width, int margin,
    const cv::Scalar &color)
{
    cv::line(img, pt_from, pt_to, color);
    cv::Point2d direction = pt_to - pt_from;
    direction /= cv::norm(direction);
    cv::Point2d op_direction{-direction.y, direction.x};

    cv::Point2d pt_arrow_end = pt_to - direction * margin;
    cv::Point2d pt_arrow_back = pt_to - direction * (len + margin);
    cv::Point2d pt_back_pos_1 = pt_arrow_back + width * len * op_direction;
    cv::Point2d pt_back_pos_2 = pt_arrow_back - width * len * op_direction;
    cv::fillConvexPoly(img, std::vector<cv::Point2i>{pt_back_pos_1, pt_arrow_end, pt_back_pos_2}, color);
}


/**
 * @brief Draw edges between selected nodes.
 * @param out_img output image matrix.
 * @param adj_list adjacency list for the whole network.
 * @param nodes indexes of the nodes that were selected.
 * @param points node positions.
 * @param params drawing parameters.
 */
void draw_edges(
    cv::Mat &out_img, const AdjacencyList &adj_list, const std::vector<int> &nodes,
    const std::vector<cv::Point2i> &points, const DrawingParameters &params)
{
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        // This is a bit nonobvious. `adj_list` is a list for all nodes in the graph,
        // while `i` is the position of node in subgraph.
        // Position of node in the whole graph is `nodes[i]`.
        for (size_t j = 0; j < adj_list[nodes[i]].size(); ++j)
        {
            // `i` is an index of a list element and index of a source element for the graph edge simultaneously.
            auto target_node_iter = std::find(nodes.begin(), nodes.end(), adj_list[nodes[i]][j]);
            if (target_node_iter == nodes.end()) continue;

            size_t target_point_index = target_node_iter - nodes.begin();
            if (target_point_index == i)
            {
                cv::circle(
                    out_img, points[i] - cv::Point2i(0, params.self_arrow_radius), params.self_arrow_radius,
                    params.edge_color);
                continue;
            }
            const cv::Point2i &point_from = points[i];
            const cv::Point2i &point_to = points[target_point_index];
            draw_simple_arrow_line(
                out_img, point_from, point_to, params.arrow_len, params.arrow_width, params.node_radius,
                params.edge_color);
        }
    }
}


/**
 * @brief Draw graph with with node names or IDs as graph legends.
 * @param graph whole network graph.
 * @param adj_list whole network adjacency list.
 */
cv::Mat draw_annotated_subgraph(
    const NetworkGraph &graph, const AdjacencyList &adj_list, const std::vector<int> &nodes,
    const std::vector<cv::Point2i> &points, const std::vector<int> &inputs, const cv::Size &img_size,
    const DrawingParameters &params = DrawingParameters{})
{
    // TODO: Draw graph of selected nodes, while the nodes are considered external if they are not included.
    cv::Mat out_img{img_size, CV_8UC3, params.back_color};
    // Draw inputs (black arrow from above).
    for (auto input : inputs)
    {
        auto target_node_iter = std::find(nodes.begin(), nodes.end(), input);
        if (target_node_iter == nodes.end()) continue;
        cv::Point2i point = points[target_node_iter - nodes.begin()];
        draw_simple_arrow_line(
            out_img, point - cv::Point2i(0, 2 * params.arrow_len + params.node_radius), point, params.arrow_len,
            params.arrow_width, params.node_radius, params.node_color);
    }

    draw_edges(out_img, adj_list, nodes, points, params);

    // Draw nodes.
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        // Drawing.
        const auto &point = points[i];
        cv::circle(
            out_img, {static_cast<int>(point.x), static_cast<int>(point.y)}, params.node_radius, params.node_color, -1);
        cv::Point2i text_start = point;
        std::string name = graph.nodes_[i].name_;
        int baseline = 0;
        auto text_size = cv::getTextSize(name, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        text_start.x = std::max(params.text_margin, text_start.x - text_size.width / 2);
        if (text_start.x + text_size.width / 2 > img_size.width - params.text_margin)
            text_start.x = img_size.width - text_size.width - params.text_margin;
        text_start.y += params.node_radius + params.text_margin + text_size.height;
        if (text_start.y > img_size.height) text_start.y = point.y - params.node_radius - params.text_margin;
        cv::putText(out_img, name, text_start, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 2);
    }
    return out_img;
}


/**
 * @brief Draw connected subgraph. It shouldn't have any connections to other parts of the graph, excluding inputs.
 * @details The function draws a vertical black arrow if a node is connected to input.
 * @param adj_list adjacency list of the whole graph.
 * @param nodes nodes that are part of the subgraph.
 * @param points node coordinates.
 * @param inputs input nodes.
 * @param img_size output image size.
 * @param params graph drawing parameters.
 * @return image of a drawn subgraph.
 */
cv::Mat draw_subgraph(
    const AdjacencyList &adj_list, const std::vector<int> &nodes, const std::vector<cv::Point2i> &points,
    const std::vector<int> &inputs, const cv::Size &img_size, const DrawingParameters &params = DrawingParameters{})
{
    cv::Mat out_img{img_size, CV_8UC3, params.back_color};
    // Draw inputs (black arrow from above).
    for (auto input : inputs)
    {
        auto target_node_iter = std::find(nodes.begin(), nodes.end(), input);
        if (target_node_iter == nodes.end()) continue;
        cv::Point2i point = points[target_node_iter - nodes.begin()];
        draw_simple_arrow_line(
            out_img, point - cv::Point2i(0, 2 * params.arrow_len + params.node_radius), point, params.arrow_len,
            params.arrow_width, params.node_radius, params.node_color);
    }

    draw_edges(out_img, adj_list, nodes, points, params);

    // Draw nodes.
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        // Drawing.
        const auto &point = points[i];
        cv::circle(
            out_img, {static_cast<int>(point.x), static_cast<int>(point.y)}, params.node_radius, params.node_color, -1);
    }
    return out_img;
}

/**
 * @brief Make a reverse adjacency list. It's used to quickly find "incoming" nodes.
 * @param adj_list regular adjacency list.
 * @return reversed adjacency list, where each node has a list of nodes it is adjacent to.
 */
AdjacencyList make_reverse_list(const AdjacencyList &adj_list)
{
    AdjacencyList rev_list;
    rev_list.resize(adj_list.size());
    for (size_t count = 0; count < adj_list.size(); ++count)
    {
        for (auto val : adj_list[count]) rev_list[val].push_back(count);
    }
    return rev_list;
}


/**
 * @brief Find an independent subgraph inside a larger graph.
 * @param adj_list adjacency list.
 * @param rev_list reversed adjacency list.
 * @param remaining_nodes set of nodes that are not currently in a subgraph.
 * @param ignore_nodes indexes of nodes to ignore.
 * @return connected subgraph of nodes.
 * @see make_reverse_list
 */
std::vector<int> find_connected_set(
    const AdjacencyList &adj_list, const AdjacencyList &rev_list, std::unordered_set<int> &remaining_nodes,
    const std::unordered_set<int> &ignore_nodes = {})
{
    auto cleaned_nodes = remaining_nodes;
    for (const auto node : ignore_nodes) cleaned_nodes.erase(node);
    if (cleaned_nodes.empty())
    {
        remaining_nodes = {};
        return {};
    }

    // Select a node.
    int node = *cleaned_nodes.begin();
    std::unordered_set<int> processed_nodes;
    std::unordered_set<int> nonproc_nodes{node};
    // Go through all its connections and add nodes to the resulting set.
    while (!nonproc_nodes.empty())
    {
        int curr_node = *nonproc_nodes.begin();
        for (const AdjacencyList &curr_list : {adj_list, rev_list})
        {
            // Add all the nodes a selected node is connected to.
            const auto &out_nodes = curr_list[curr_node];
            for (auto node_out : out_nodes)
            {
                int n = static_cast<int>(node_out);
                if (n == curr_node) continue;
                if (processed_nodes.find(n) != processed_nodes.end()) continue;  // Node has already been processed.
                if (ignore_nodes.find(n) != ignore_nodes.end()) continue;        // Node is explicitly excluded.
                nonproc_nodes.insert(n);
            }
        }
        // Move current node from unprocessed to processed.
        nonproc_nodes.erase(curr_node);
        processed_nodes.insert(curr_node);
    }

    for (int node_id : processed_nodes) remaining_nodes.erase(node_id);

    // Return connected nodes as a sorted vector.
    std::vector<int> result;
    result.resize(processed_nodes.size());
    std::copy(processed_nodes.begin(), processed_nodes.end(), result.begin());
    std::sort(result.begin(), result.end());
    return result;
}


/**
 * @brief Find all independent components inside a graph.
 * @param graph network graph.
 */
std::vector<std::vector<int>> divide_graph_by_connectivity(const NetworkGraph &graph)
{
    AdjacencyList adj_list = build_adjacency_list(graph);
    AdjacencyList rev_list = make_reverse_list(adj_list);
    std::unordered_set<int> remaining_nodes;
    std::vector<std::vector<int>> connected_sets;
    std::unordered_set<int> ignored_nodes{static_cast<int>(adj_list.size() - 1)};
    for (int i = 0; i < adj_list.size(); ++i) remaining_nodes.insert(i);

    remaining_nodes.erase(static_cast<int>(adj_list.size()) - 1);  // delete input node
    while (!remaining_nodes.empty())
    {
        connected_sets.push_back(find_connected_set(adj_list, rev_list, remaining_nodes, ignored_nodes));
    }

    return connected_sets;
}


/**
 * @brief Print network subset description.
 * @param adj_list adjacency list.
 * @param rev_list reversed adjacency list.
 * @param nodes nodes from a connected subset.
 * @see build_adjacency_list, make_reverse_list, and find_connected_set.
 */
void print_connected_subset(
    const NetworkGraph &graph, const AdjacencyList &adj_list, const AdjacencyList &rev_list,
    const std::vector<int> &nodes)
{
    for (auto node : nodes)
    {
        std::cout << "Population #" << node << " of size " << graph.nodes_[node].size_ << ": receive from";
        for (auto node_from : rev_list[node])
            if (node_from == adj_list.size() - 1)
                std::cout << " #INPUT";
            else
                std::cout << " #" << node_from;
        std::cout << "; send to";
        for (auto node_to : adj_list[node]) std::cout << " #" << node_to;
        std::cout << std::endl;
    }
}


/**
 * @brief Print all connected subsets descriptions.
 * @param graph network graph.
 */
void print_network_description(const NetworkGraph &graph)
{
    AdjacencyList adj_list = build_adjacency_list(graph);
    AdjacencyList rev_list = make_reverse_list(adj_list);
    auto connected_subsets = divide_graph_by_connectivity(graph);
    for (const auto &subset : connected_subsets)
    {
        print_connected_subset(graph, adj_list, rev_list, subset);
        std::cout << std::endl;
    }
}


/**
 * @brief Show the process of subgraph adjustment.
 * @param graph full network graph.
 * @param nodes all nodes that are contained in a subgraph.
 * @param screen_size output window size.
 * @param margin margin size in pixels.
 */
void position_network_test(
    const NetworkGraph &graph, const std::vector<int> &nodes, const cv::Size &screen_size, int margin)
{
    cv::theRNG().state = std::time(nullptr);
    AdjacencyList adj_list = build_adjacency_list(graph);
    VisualGraph vis_graph(nodes, adj_list);
    int key = 0;

    // Create inputs.
    std::vector<int> inputs;
    inputs.reserve(adj_list.back().size());
    std::transform(
        adj_list.back().begin(), adj_list.back().end(), std::back_inserter(inputs),
        [](size_t v) { return static_cast<int>(v); });
    // for (auto v : adj_list.back()) inputs.push_back(static_cast<int>(v));

    while (key != 27)
    {
        auto points = vis_graph.scale_graph(screen_size, margin);
        cv::Mat img = draw_annotated_subgraph(graph, adj_list, nodes, points, inputs, screen_size);
        cv::imshow("Graph", img);
        key = cv::waitKey(50) & 255;
        vis_graph.iterate(1);
    }
}


/**
 * @brief Calculate positions of nodes.
 * @param graph full network graph.
 * @param nodes all nodes that are contained in a subgraph.
 * @param screen_size output window size.
 * @param margin margin size in pixels.
 * @param num_iterations number of iterations for graph positioning algorithm.
 * @return node coordinates.
 */
std::vector<cv::Point2i> position_network(
    const NetworkGraph &graph, const std::vector<int> &nodes, const cv::Size &screen_size, int margin,
    int num_iterations)
{
    VisualGraph vis_graph(nodes, build_adjacency_list(graph));
    vis_graph.iterate(num_iterations);
    auto result = vis_graph.scale_graph(screen_size, margin);
    return result;
}

// TODO: Draw network as a set of subgraphs.
