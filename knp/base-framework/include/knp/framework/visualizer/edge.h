/**
 * @file edge.h
 * @brief Edge of a network graph.
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

#include <knp/core/uid.h>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Edge kind of a projection.
 */
constexpr char projection_edge_type[]{"Projection"};


/**
 * @brief Network edge representation.
 *
 * @details Stores projection information including size, connection UIDs, UID, name and type.
 * The class is stored as an edge property in Boost Graph.
 */
class Edge
{
public:
    /**
     * @brief Default constructor. Boost Graph needs it for edge properties.
     */
    Edge() = default;

    /**
     * @brief Construct an edge.
     *
     * @param uid entity UID.
     * @param name entity name.
     * @param index_from source UID, `std::nullopt` for an invisible source.
     * @param index_to target UID, `std::nullopt` for an invisible target.
     * @param size number of synapses, if known.
     * @param type index of the synapse type in `knp::core::AllProjectionsVariant`, if known.
     * @param comment comment written to the edge.
     */
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

    /**
     * @brief Get edge UID.
     *
     * @return edge UID.
     */
    [[nodiscard]] const knp::core::UID &get_uid() const { return uid_; }

    /**
     * @brief Get edge name.
     *
     * @return edge name.
     */
    [[nodiscard]] const std::string &get_name() const { return name_; }

    /**
     * @brief Get source UID.
     *
     * @return source UID, or `std::nullopt` for an invisible source.
     */
    [[nodiscard]] const std::optional<knp::core::UID> &get_index_from() const { return index_from_; }

    /**
     * @brief Set source UID.
     *
     * @param index_from source UID.
     */
    void set_index_from(const knp::core::UID &index_from) { index_from_ = index_from; }

    /**
     * @brief Get target UID.
     *
     * @return target UID, or `std::nullopt` for an invisible target.
     */
    [[nodiscard]] const std::optional<knp::core::UID> &get_index_to() const { return index_to_; }

    /**
     * @brief Set target UID.
     *
     * @param index_to target UID.
     */
    void set_index_to(const knp::core::UID &index_to) { index_to_ = index_to; }

    /**
     * @brief Get number of synapses.
     *
     * @return number of synapses, or `std::nullopt` if it is unknown.
     */
    [[nodiscard]] const std::optional<size_t> &get_size() const { return size_; }

    /**
     * @brief Get synapse type.
     *
     * @return index of the synapse type in `knp::core::AllProjectionsVariant`, or `std::nullopt` if it is unknown.
     */
    [[nodiscard]] const std::optional<size_t> &get_type() const { return type_; }

    /**
     * @brief Get edge comment.
     *
     * @return edge comment, or `std::nullopt` if there is none.
     */
    [[nodiscard]] const std::optional<std::string> &get_comment() const { return comment_; }

    /**
     * @brief Get edge kind as it is written to the edge label.
     *
     * @return edge kind.
     */
    [[nodiscard]] const std::string &get_edge_type() const { return edge_type_; }

    /**
     * @brief Set edge kind as it is written to the edge label.
     *
     * @param edge_type edge kind.
     */
    void set_edge_type(std::string edge_type) { edge_type_ = std::move(edge_type); }

    /**
     * @brief Check that the edge is described by the network.
     *
     * @return `true` if the edge is described by the network.
     */
    [[nodiscard]] bool is_static() const { return is_static_; }

    /**
     * @brief Set a flag showing that the edge is described by the network.
     *
     * @param is_static `true` if the edge is described by the network.
     */
    void set_static(bool is_static) { is_static_ = is_static; }

    /**
     * @brief Check that the edge is confirmed by message bus subscriptions.
     *
     * @return `true` if the edge is confirmed by message bus subscriptions.
     */
    [[nodiscard]] bool is_dynamic() const { return is_dynamic_; }

    /**
     * @brief Set a flag showing that the edge is confirmed by message bus subscriptions.
     *
     * @param is_dynamic `true` if the edge is confirmed by message bus subscriptions.
     */
    void set_dynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }

private:
    knp::core::UID uid_;
    std::string name_;
    std::optional<knp::core::UID> index_from_;  // Source UID, `std::nullopt` for an invisible source.
    std::optional<knp::core::UID> index_to_;    // Target UID, `std::nullopt` for an invisible target.
    std::optional<size_t> size_;
    std::optional<size_t> type_;  // Index of the synapse type in `knp::core::AllProjectionsVariant`.
    std::optional<std::string> comment_;

    /**
     * @brief Edge kind as it is written to the edge label: `projection_edge_type` or "Spike Message".
     */
    std::string edge_type_ = projection_edge_type;

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
};

}  // namespace knp::framework::visualizer
