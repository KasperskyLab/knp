/**
 * @file graph_physics.h
 * @brief Building a graph structure.
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

#include <vector>

#include <opencv2/core.hpp>


/**
 * @brief A point that has position and velocity.
 */
struct PhysicsPoint
{
    /**
     * @brief Point index.
     */
    // cppcheck-suppress unusedStructMember
    int index_;

    /**
     * @brief Point position.
     */
    // cppcheck-suppress unusedStructMember
    cv::Vec2d pos_;

    /**
     * @brief Point velocity.
     */
    // cppcheck-suppress unusedStructMember
    cv::Vec2d vel_;
};


/**
 * Graph description with additional parameters used for visualization.
 */
class VisualGraph
{
public:
    /**
     * @brief Construct graph visualizer.
     * @param nodes subgraph node indexes.
     * @param adj_list whole graph adjacency list.
     */
    explicit VisualGraph(const std::vector<int> &nodes, const std::vector<std::vector<size_t>> &adj_list);

    /**
     * @brief Calculate forces, modify point velocities, and move points around.
     */
    void iterate();

    /**
     * @brief Iterate N times.
     * @param n number of iterations.
     */
    void iterate(int n);

    /**
     * @brief Fit graph into rectangle, get point coordinates for the image.
     * @param screen_size output rectangle size.
     * @param margin margin sizes.
     */
    [[nodiscard]] std::vector<cv::Point2i> scale_graph(const cv::Size &screen_size, int margin) const;

    /**
     * @brief Get point positions.
     * @return point positions.
     */
    [[nodiscard]] std::vector<cv::Point2d> get_positions() const;

private:
    // Data.
    // cppcheck-suppress unusedStructMember
    std::vector<std::vector<size_t>> base_graph_;
    // Edges. If `true`, then there is an edge.
    // cppcheck-suppress unusedStructMember
    std::vector<std::vector<bool>> edges_mat_;
    // Positions and velocities of the nodes.
    // cppcheck-suppress unusedStructMember
    std::vector<PhysicsPoint> points_;

    // Hyperparameters.
    // cppcheck-suppress unusedStructMember
    double spring_strength_ = 1.0f;
    // cppcheck-suppress unusedStructMember
    double spring_len_ = 1.0f;
    // cppcheck-suppress unusedStructMember
    double repel_coeff_ = 0.3f;
    // cppcheck-suppress unusedStructMember
    double resistance_ = 1.0f;

    // Calculate force from one graph point to another. Force consists of "repel", "spring" and "resistance" components.
    [[nodiscard]] cv::Vec2d get_force(const PhysicsPoint &target, const PhysicsPoint &influence, bool has_edge) const;
};
