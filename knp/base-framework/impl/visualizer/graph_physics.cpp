/**
 * @file graph_physics.cpp
 * @brief Implementing a graph physics.
 * @date 21.08.2024
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

#include "graph_physics.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>

#include <opencv2/core.hpp>


// Construct a graph class from an adjacency list.
VisualGraph::VisualGraph(const std::vector<int> &nodes, const std::vector<std::vector<size_t>> &adj_list)
    : base_graph_(adj_list)
{
    std::unordered_map<size_t, size_t> point_by_original_index;
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        PhysicsPoint point{nodes[i], {cv::randu<double>(), cv::randu<double>()}, {0.0, 0.0}};
        points_.push_back(std::move(point));
        point_by_original_index.insert({points_[i].index_, i});
    }

    edges_mat_.resize(nodes.size());
    for (auto &edge : edges_mat_) edge.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        for (size_t index : adj_list[points_[i].index_]) edges_mat_[i][point_by_original_index[index]] = true;
    }
}


// We show graph using a following physical model: all nodes repel each other, all connected nodes are connected
// as if by springs of a fixed length. There is also a drag force to dampen oscillations, it's calculated
// independently. This function calculates forces between two graph nodes.
cv::Vec2d VisualGraph::get_force(const PhysicsPoint &target, const PhysicsPoint &influence, bool has_edge) const
{
    double very_small_number = 1e-5;
    cv::Vec2d dist = target.pos_ - influence.pos_;
    double len = cv::norm(dist);
    cv::Vec2d force = 0;
    if (has_edge) force += -dist * (len - spring_len_) / std::max(len, very_small_number) * spring_strength_;
    force += dist / (len * len * len + very_small_number) * repel_coeff_;
    return force;
}


// A single iteration recalculates forces and velocities then moves the graph points according to their corresponding
// velocities.
void VisualGraph::iterate()
{
    double dt = 0.2;
    // Calculate forces.
    std::vector<cv::Vec2d> forces;
    forces.resize(points_.size(), {0.0, 0.0});
    for (size_t i = 0; i < points_.size(); ++i)
    {
        for (size_t j = 0; j < points_.size(); ++j)
        {
            if (i == j) continue;
            auto force = get_force(points_[i], points_[j], edges_mat_[i][j] || edges_mat_[j][i]);
            forces[i] += force;
        }
        // Drag.
        forces[i] += -points_[i].vel_ * resistance_;
    }
    // Update velocities and positions.
    for (size_t i = 0; i < points_.size(); ++i)
    {
        points_[i].vel_ += forces[i] * dt;
        points_[i].pos_ += points_[i].vel_ * dt;
    }
}


// Iterate N times.
void VisualGraph::iterate(int n)
{
    for (int i = 0; i < n; ++i) iterate();
}


// Get positions for all graph points.
[[nodiscard]] std::vector<cv::Point2d> VisualGraph::get_positions() const
{
    std::vector<cv::Point2d> result;
    result.reserve(points_.size());
    std::transform(
        points_.begin(), points_.end(), std::back_inserter(result),
        [](const auto &v) { return cv::Point2d(v.pos_[0], v.pos_[1]); });
    return result;
}


// Rescale a graph to make it fit inside a rectangle.
std::vector<cv::Point2i> VisualGraph::scale_graph(const cv::Size &screen_size, int margin) const
{
    const double small_num = 1e-5;
    cv::Point2d low_point{std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    cv::Point2d high_point{std::numeric_limits<double>::min(), std::numeric_limits<double>::min()};
    cv::Size graph_max_size = screen_size - cv::Size{2 * margin, 2 * margin};
    auto points = get_positions();

    // TODO: Maybe do a rotation here later, probably using cv::minAreaRect.
    for (const auto &point : points)
    {
        low_point.x = std::min(low_point.x, point.x);
        low_point.y = std::min(low_point.y, point.y);
        high_point.x = std::max(high_point.x, point.x);
        high_point.y = std::max(high_point.y, point.y);
    }

    cv::Size2d actual_size(cv::Point2d{high_point.x - low_point.x, high_point.y - low_point.y});
    double scale_x = static_cast<double>(graph_max_size.width) / (actual_size.width + small_num);
    double scale_y = static_cast<double>(graph_max_size.height) / (actual_size.height + small_num);

    // Rescale to fit dimensions.
    double scale = std::min(scale_x, scale_y);
    std::vector<cv::Point2i> result;
    result.reserve(points.size());
    for (const auto &point : points)
    {
        cv::Point2l pt(
            std::lround((point.x - low_point.x) * scale) + margin,
            std::lround((point.y - low_point.y) * scale) + margin);
        result.push_back(std::move(pt));
    }
    return result;
}
