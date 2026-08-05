#pragma once
// #include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C"
{
#include <graphviz/gvc.h>
}

/**
 * @brief Get bus messages from backend subscriptions
 *
 * @param backend shared pointer to backend
 *
 * @return tuple of spike messages and synaptic messages
 */
std::tuple<
    std::vector<std::pair<knp::core::UID, knp::core::UID>>, std::vector<std::pair<knp::core::UID, knp::core::UID>>>
get_bus(std::shared_ptr<knp::core::Backend>& backend);

/**
 * @brief Get entity name for any network object. If there is no name it's constructed from UID.
 *
 * @param pop population object
 *
 * @return string name of the population
 */
template <class Entity>
std::string get_population_name(const Entity& pop)
{
    const size_t uid_part_size = 8;
    knp::core::TagMap tags = std::visit([](const auto& p) { return p.get_tags(); }, pop);
    knp::core::UID uid = std::visit([](const auto& p) { return p.get_uid(); }, pop);
    std::string name;
    auto tag = tags.get_tag("name");
    if (tag.has_value())
    {
        try
        {
            name = std::any_cast<std::string>(tag);
        }
        catch (std::bad_any_cast& exc)
        {
            SPDLOG_WARN("Wrong name tag type.");
            name = std::string{uid}.substr(0, uid_part_size);
        }
    }
    if (name.empty()) name = std::string{uid}.substr(0, uid_part_size);
    return name;
}

/**
 * @brief Get node name by UID from network graph nodes
 *
 * @param node_uid UID of the node
 * @param nodes vector of network graph nodes
 *
 * @return string node name
 */
std::string get_node_name(knp::core::UID node_uid, std::vector<knp::framework::NetworkGraph::Node> const& nodes);

/**
 * @brief Get graph node by UID from network graph nodes
 *
 * @param node_uid UID of the node
 * @param nodes vector of network graph nodes
 *
 * @return NetworkGraph::Node object
 */
knp::framework::NetworkGraph::Node get_graph_node_by_uid(
    knp::core::UID node_uid, std::vector<knp::framework::NetworkGraph::Node> const& nodes);

/**
 * @brief Get projection size by UID from network graph edges
 *
 * @param edges vector of network graph edges
 * @param edge_uid UID of the edge
 *
 * @return string size of the edge
 */
std::string get_proj_size(std::vector<knp::framework::NetworkGraph::Edge> const& edges, const knp::core::UID& edge_uid);
