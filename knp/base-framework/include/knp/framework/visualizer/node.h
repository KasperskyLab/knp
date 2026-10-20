/**
 * @file node.h
 * @brief Node of a network graph.
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
 * @brief Node kind of a population described by the network.
 */
constexpr char population_node_type[]{"Population"};


/**
 * @brief Network node representation.
 *
 * @details Stores population information including size, UID, name and type.
 * The class is stored as a vertex property in Boost Graph.
 */
class Node
{
public:
    /**
     * @brief Default constructor. Boost Graph needs it for vertex properties.
     */
    Node() = default;

    /**
     * @brief Construct a node.
     *
     * @param uid entity UID.
     * @param name entity name.
     * @param size number of neurons, if known.
     * @param type index of the neuron type in `knp::core::AllPopulationsVariant`, if known.
     * @param comment comment written to the node.
     */
    Node(
        knp::core::UID uid, std::string name, std::optional<size_t> size, std::optional<size_t> type,
        std::optional<std::string> comment)
        : uid_(uid), name_(std::move(name)), size_(size), type_(type), comment_(std::move(comment))
    {
    }

    /**
     * @brief Get node UID.
     *
     * @return node UID.
     */
    [[nodiscard]] const knp::core::UID &get_uid() const { return uid_; }

    /**
     * @brief Get node name.
     *
     * @return node name.
     */
    [[nodiscard]] const std::string &get_name() const { return name_; }

    /**
     * @brief Get number of neurons.
     *
     * @return number of neurons, or `std::nullopt` if it is unknown.
     */
    [[nodiscard]] const std::optional<size_t> &get_size() const { return size_; }

    /**
     * @brief Get neuron type.
     *
     * @return index of the neuron type in `knp::core::AllPopulationsVariant`, or `std::nullopt` if it is unknown.
     */
    [[nodiscard]] const std::optional<size_t> &get_type() const { return type_; }

    /**
     * @brief Get node comment.
     *
     * @return node comment, or `std::nullopt` if there is none.
     */
    [[nodiscard]] const std::optional<std::string> &get_comment() const { return comment_; }

    /**
     * @brief Get node kind as it is written to the node label.
     *
     * @return node kind.
     */
    [[nodiscard]] const std::string &get_node_type() const { return node_type_; }

    /**
     * @brief Set node kind as it is written to the node label.
     *
     * @param node_type node kind.
     */
    void set_node_type(std::string node_type) { node_type_ = std::move(node_type); }

    /**
     * @brief Check that the node is described by the network.
     *
     * @return `true` if the node is described by the network, `false` if it is known from the message bus only.
     */
    [[nodiscard]] bool is_static() const { return is_static_; }

    /**
     * @brief Set a flag showing that the node is described by the network.
     *
     * @param is_static `true` if the node is described by the network.
     */
    void set_static(bool is_static) { is_static_ = is_static; }

    /**
     * @brief Check that the node is invisible.
     *
     * @return `true` for a placeholder of an edge without source or destination.
     */
    [[nodiscard]] bool is_invisible() const { return is_invisible_; }

    /**
     * @brief Set a flag showing that the node is invisible.
     *
     * @param is_invisible `true` for a placeholder of an edge without source or destination.
     */
    void set_invisible(bool is_invisible) { is_invisible_ = is_invisible; }

private:
    knp::core::UID uid_;
    std::string name_;
    std::optional<size_t> size_;
    std::optional<size_t> type_;  // Index of the neuron type in `knp::core::AllPopulationsVariant`.
    std::optional<std::string> comment_;

    /**
     * @brief Node kind as it is written to the node label.
     *
     * @details `population_node_type` for a population of the network. A node that is not described by the network
     * gets the kind given by the calling code, or "External" if the calling code does not describe it.
     */
    std::string node_type_ = population_node_type;

    /**
     * @brief A flag showing a dynamic or static node.
     *
     * @details `true` if the node is described by the network, `false` if it is known from the message bus only.
     */
    bool is_static_ = true;

    /**
     * @brief A flag showing a visible or invisible node.
     *
     * @details It is needed for drawing edges without source or destination.
     */
    bool is_invisible_ = false;
};

}  // namespace knp::framework::visualizer
