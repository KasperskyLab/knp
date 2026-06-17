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

#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "graph_physics.h"


// TODO: Draw network as a set of subgraphs.
namespace knp::framework
{

/**
 * @brief Adjacency list is a representation of a graph where there is a list of adjacent nodes for each node.
 * @see https://en.wikipedia.org/wiki/Adjacency_list.
 */
using AdjacencyList = std::vector<std::vector<size_t>>;


namespace
{
const size_t unlimited_synapses = std::numeric_limits<size_t>::max();
}


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
        std::string name = graph.nodes_[nodes[i]].name_;
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
    for (size_t i = 0; i < adj_list.size(); ++i) remaining_nodes.insert(i);

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
 * @brief Show the whole network graph as one static image.
 * @param graph full network graph.
 * @param screen_size output window size.
 * @param margin margin size in pixels.
 * @param num_iterations number of iterations for graph positioning algorithm.
 */
void show_network(const NetworkGraph &graph, const cv::Size &screen_size, int margin, int num_iterations)
{
    AdjacencyList adj_list = build_adjacency_list(graph);
    std::vector<int> nodes(graph.nodes_.size());
    std::iota(nodes.begin(), nodes.end(), 0);

    std::vector<int> inputs;
    inputs.reserve(adj_list.back().size());
    std::transform(
        adj_list.back().begin(), adj_list.back().end(), std::back_inserter(inputs),
        [](size_t v) { return static_cast<int>(v); });

    VisualGraph vis_graph(nodes, adj_list);
    vis_graph.iterate(num_iterations);
    const auto points = vis_graph.scale_graph(screen_size, margin);
    const cv::Mat img = draw_annotated_subgraph(graph, adj_list, nodes, points, inputs, screen_size);
    cv::imshow("Graph", img);
    cv::waitKey(0);
}


struct NeuronGroup
{
    std::string key_;
    std::string label_;
    size_t size_ = 0;
    bool external_ = false;
    cv::Rect area_;
    std::vector<cv::Point2i> points_;
};


struct NeuronProjection
{
    std::string source_key_;
    std::string target_key_;
    std::string label_;
    size_t synapse_count_ = 0;
    size_t drawn_count_ = 0;
};


std::string uid_key(const knp::core::UID &uid)
{
    return static_cast<std::string>(uid);
}


std::string uid_label(const knp::core::UID &uid)
{
    constexpr size_t uid_part_size = 8;
    return uid_key(uid).substr(0, uid_part_size);
}


void add_external_group(
    std::vector<NeuronGroup> &groups, const std::string &key, const std::string &label, size_t size)
{
    const auto group_iter = std::find_if(groups.begin(), groups.end(), [&key](const auto &group) {
        return group.key_ == key;
    });
    if (group_iter == groups.end()) groups.push_back(NeuronGroup{key, label, size, true});
}


std::string find_population_key_by_size(
    const std::vector<NeuronGroup> &groups, size_t source_size, const std::string &target_key)
{
    std::string result;
    for (const auto &group : groups)
    {
        if (group.external_ || group.key_ == target_key || group.size_ != source_size) continue;
        if (!result.empty()) return {};
        result = group.key_;
    }
    return result;
}


template <typename Projection>
std::string resolve_source_group(
    const Projection &projection, std::vector<NeuronGroup> &groups, size_t source_size,
    const std::string &target_key)
{
    const auto source_uid = projection.get_presynaptic();
    if (source_uid) return uid_key(source_uid);

    // WTA-backed projections are saved with an empty presynaptic UID. If their source neuron indexes exactly match
    // one existing population, use that population as a visual source so neuron-level diagrams stay understandable.
    const std::string inferred_key = find_population_key_by_size(groups, source_size, target_key);
    if (!inferred_key.empty()) return inferred_key;

    const std::string external_key = "external:" + uid_key(projection.get_uid());
    add_external_group(groups, external_key, "INPUT[" + std::to_string(source_size) + "]", source_size);
    return external_key;
}


template <typename Projection>
size_t get_projection_source_size(const Projection &projection)
{
    size_t source_size = 0;
    for (const auto &synapse : projection)
    {
        source_size = std::max(source_size, std::get<knp::core::source_neuron_id>(synapse) + 1);
    }
    return source_size;
}


std::unordered_map<std::string, size_t> build_group_index(const std::vector<NeuronGroup> &groups)
{
    std::unordered_map<std::string, size_t> group_index;
    for (size_t i = 0; i < groups.size(); ++i) group_index[groups[i].key_] = i;
    return group_index;
}


void layout_neuron_groups(std::vector<NeuronGroup> &groups, const cv::Size &screen_size)
{
    constexpr int top_margin = 70;
    constexpr int bottom_margin = 35;
    constexpr int side_margin = 45;
    constexpr int group_gap = 30;

    const int group_count = static_cast<int>(groups.size());
    if (group_count == 0) return;

    const int available_width = std::max(1, screen_size.width - 2 * side_margin - group_gap * (group_count - 1));
    const int group_width = std::max(1, available_width / group_count);
    const int group_height = std::max(1, screen_size.height - top_margin - bottom_margin);

    for (int i = 0; i < group_count; ++i)
    {
        auto &group = groups[i];
        group.area_ = cv::Rect(side_margin + i * (group_width + group_gap), top_margin, group_width, group_height);

        const double ratio = static_cast<double>(group.area_.width) / static_cast<double>(group.area_.height);
        const int columns = std::max(1, static_cast<int>(std::ceil(std::sqrt(group.size_ * ratio))));
        const int rows = std::max(1, static_cast<int>(std::ceil(static_cast<double>(group.size_) / columns)));
        group.points_.clear();
        group.points_.reserve(group.size_);

        for (size_t neuron_index = 0; neuron_index < group.size_; ++neuron_index)
        {
            const int column = static_cast<int>(neuron_index % columns);
            const int row = static_cast<int>(neuron_index / columns);
            const int x = group.area_.x + (2 * column + 1) * group.area_.width / (2 * columns);
            const int y = group.area_.y + (2 * row + 1) * group.area_.height / (2 * rows);
            group.points_.push_back({x, y});
        }
    }
}


template <typename Projection>
NeuronProjection draw_projection_neurons(
    cv::Mat &image, const Projection &projection, const std::unordered_map<std::string, size_t> &group_index,
    const std::vector<NeuronGroup> &groups, const std::string &source_key, const std::string &target_key,
    size_t max_synapses_per_projection)
{
    NeuronProjection projection_info{
        source_key, target_key, uid_label(projection.get_uid()), projection.size(), 0};
    if (projection.size() == 0) return projection_info;

    const auto source_group_iter = group_index.find(source_key);
    const auto target_group_iter = group_index.find(target_key);
    if (source_group_iter == group_index.end() || target_group_iter == group_index.end()) return projection_info;

    const auto &source_group = groups[source_group_iter->second];
    const auto &target_group = groups[target_group_iter->second];
    const size_t max_lines = max_synapses_per_projection == 0 ? unlimited_synapses : max_synapses_per_projection;
    const size_t step = max_lines == unlimited_synapses ? 1 : std::max<size_t>(1, projection.size() / max_lines);
    const cv::Scalar line_color = source_group.external_ ? cv::Scalar{160, 160, 160} : cv::Scalar{190, 110, 40};

    for (size_t synapse_index = 0; synapse_index < projection.size(); synapse_index += step)
    {
        if (projection_info.drawn_count_ >= max_lines) break;
        const auto &synapse = projection[synapse_index];
        const size_t source_neuron = std::get<knp::core::source_neuron_id>(synapse);
        const size_t target_neuron = std::get<knp::core::target_neuron_id>(synapse);
        if (source_neuron >= source_group.points_.size() || target_neuron >= target_group.points_.size()) continue;
        cv::line(
            image, source_group.points_[source_neuron], target_group.points_[target_neuron], line_color, 1,
            cv::LINE_AA);
        ++projection_info.drawn_count_;
    }

    return projection_info;
}


void draw_neuron_groups(cv::Mat &image, const std::vector<NeuronGroup> &groups)
{
    const size_t max_group_size =
        std::accumulate(groups.begin(), groups.end(), size_t{0}, [](size_t current, const auto &group) {
            return std::max(current, group.size_);
        });
    const int radius = max_group_size > 500 ? 1 : (max_group_size > 100 ? 2 : 4);

    for (const auto &group : groups)
    {
        const cv::Scalar color = group.external_ ? cv::Scalar{80, 130, 80} : cv::Scalar{0, 0, 0};
        cv::rectangle(image, group.area_, cv::Scalar{220, 220, 220}, 1);

        for (const auto &point : group.points_) cv::circle(image, point, radius, color, -1, cv::LINE_AA);

        std::ostringstream label;
        label << group.label_ << " (" << group.size_ << ")";
        cv::putText(
            image, label.str(), {group.area_.x, group.area_.y - 18}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar{0, 0, 0}, 2);
    }
}


void draw_projection_labels(
    cv::Mat &image, const std::vector<NeuronGroup> &groups, const std::vector<NeuronProjection> &projections)
{
    const auto group_index = build_group_index(groups);
    std::map<std::pair<std::string, std::string>, size_t> label_offsets;

    for (const auto &projection : projections)
    {
        const auto source_group_iter = group_index.find(projection.source_key_);
        const auto target_group_iter = group_index.find(projection.target_key_);
        if (source_group_iter == group_index.end() || target_group_iter == group_index.end()) continue;

        const auto &source_group = groups[source_group_iter->second];
        const auto &target_group = groups[target_group_iter->second];
        const auto key = std::make_pair(projection.source_key_, projection.target_key_);
        const int offset = static_cast<int>(label_offsets[key]++) * 18;
        const int x = (source_group.area_.x + source_group.area_.width + target_group.area_.x) / 2 - 55;
        const int y = 32 + offset;

        std::ostringstream label;
        label << projection.synapse_count_ << " syn";
        if (projection.drawn_count_ < projection.synapse_count_)
            label << ", " << projection.drawn_count_ << " shown";
        cv::putText(image, label.str(), {x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar{70, 70, 70}, 1);
    }
}


/**
 * @brief Show individual neurons and sampled synaptic connections as one static image.
 * @param network source network.
 * @param screen_size output window size.
 * @param max_synapses_per_projection maximum number of synapse lines to draw for each projection.
 */
void show_neuron_network(const Network &network, const cv::Size &screen_size, size_t max_synapses_per_projection)
{
    std::vector<NeuronGroup> groups;
    groups.reserve(network.get_populations().size() + network.get_projections().size());

    for (const auto &population : network.get_populations())
    {
        const auto uid = std::visit([](const auto &pop) { return pop.get_uid(); }, population);
        const size_t size = std::visit([](const auto &pop) { return pop.size(); }, population);
        groups.push_back(NeuronGroup{uid_key(uid), get_name(population), size, false});
    }

    std::stable_sort(groups.begin(), groups.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.external_ != rhs.external_) return lhs.external_ > rhs.external_;
        return lhs.size_ > rhs.size_;
    });

    std::vector<NeuronProjection> projection_infos;
    std::vector<std::tuple<const core::AllProjectionsVariant *, std::string, std::string>> resolved_projections;
    resolved_projections.reserve(network.get_projections().size());

    for (const auto &projection_variant : network.get_projections())
    {
        std::visit(
            [&groups, &projection_variant, &resolved_projections](const auto &projection) {
                const std::string target_key = uid_key(projection.get_postsynaptic());
                const size_t source_size = get_projection_source_size(projection);
                const std::string source_key = resolve_source_group(projection, groups, source_size, target_key);
                resolved_projections.emplace_back(&projection_variant, source_key, target_key);
            },
            projection_variant);
    }

    std::stable_sort(groups.begin(), groups.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.external_ != rhs.external_) return lhs.external_ > rhs.external_;
        return lhs.size_ > rhs.size_;
    });
    layout_neuron_groups(groups, screen_size);

    cv::Mat image{screen_size, CV_8UC3, cv::Scalar{255, 255, 255}};
    const auto group_index = build_group_index(groups);

    for (const auto &[projection_variant, source_key, target_key] : resolved_projections)
    {
        std::visit(
            [&image, &group_index, &groups, &source_key, &target_key, max_synapses_per_projection, &projection_infos](
                const auto &projection) {
                projection_infos.push_back(draw_projection_neurons(
                    image, projection, group_index, groups, source_key, target_key, max_synapses_per_projection));
            },
            *projection_variant);
    }

    draw_neuron_groups(image, groups);
    draw_projection_labels(image, groups, projection_infos);
    cv::imshow("Neuron graph", image);
    cv::waitKey(0);
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

}  // namespace knp::framework
