#include "utils_graph.h"

#include <knp/framework/network.h>
#include <knp/framework/visualizer/visualize_network.h>

#include <spdlog/spdlog.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>


/**
 * @brief Get bus messages from backend subscriptions
 *
 * @param backend shared pointer to backend
 *
 * @return tuple of spike messages and synaptic messages
 */
using SpikeMessages = std::vector<std::pair<knp::core::UID, knp::core::UID>>;
using SynapticMessages = std::vector<std::pair<knp::core::UID, knp::core::UID>>;
using BusMessages = std::tuple<SpikeMessages, SynapticMessages>;

BusMessages get_bus(std::shared_ptr<knp::core::Backend>& backend)
{
    const auto subs = backend->get_message_endpoint().get_endpoint_subscriptions();

    constexpr size_t SPIKE_IDX = 0;            // must be 0
    constexpr size_t SYNAPTIC_IMPACT_IDX = 1;  // must be 1

    SpikeMessages spike_messages;        // [sender, receiver]
    SynapticMessages synaptic_messages;  // [sender, receiver]


    // take from bus nessosary maps
    for (const auto& [key, sub_variant] : subs)
    {
        // const auto& [type_idx, receiver_uid] = key;
        const auto& type_idx = key.first;
        const auto& receiver_uid = key.second;
        std::visit(
            [&](const auto& sub)
            {
                const auto& senders = sub.get_senders();
                if (senders.empty())
                {
                    SPDLOG_WARN("Empty senders list for subscription");
                    return;
                }


                if (type_idx == SPIKE_IDX)
                {  // receiver: Projection,      sender: Population // 0 -->
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(spike_messages),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
                else if (type_idx == SYNAPTIC_IMPACT_IDX)
                {  // receiver: Population,      sender: Projection //  --> 0
                    std::transform(
                        senders.begin(), senders.end(), std::back_inserter(synaptic_messages),
                        [&receiver_uid](const auto& sender) { return std::make_pair(sender, receiver_uid); });
                }
            },
            sub_variant);
    }

    return {spike_messages, synaptic_messages};
}


/**
 * @brief Get node name by UID from network graph nodes
 *
 * @param node_uid UID of the node
 * @param nodes vector of network graph nodes
 *
 * @return string node name
 */
std::string get_node_name(knp::core::UID node_uid, std::vector<knp::framework::NetworkGraph::Node> const& nodes)
{
    std::string node_name = std::string(node_uid).substr(0, 8);
    // nodes from netGraph
    auto node_it =
        std::find_if(nodes.begin(), nodes.end(), [&node_uid](const auto& node) { return node.uid_ == node_uid; });
    if (node_it != nodes.end())
    {
        node_name = node_it->name_;
    }
    return node_name;
}

/**
 * @brief Get graph node by UID from network graph nodes
 *
 * @param node_uid UID of the node
 * @param nodes vector of network graph nodes
 *
 * @return NetworkGraph::Node object
 */
knp::framework::NetworkGraph::Node get_graph_node_by_uid(
    knp::core::UID node_uid, std::vector<knp::framework::NetworkGraph::Node> const& nodes)
{
    auto node_it =
        std::find_if(nodes.begin(), nodes.end(), [&node_uid](const auto& node) { return node.uid_ == node_uid; });
    if (node_it != nodes.end())
    {
        return *node_it;
    }

    SPDLOG_WARN("Node with UID {} not found in graph", std::string(node_uid));
    return knp::framework::NetworkGraph::Node{};
}

/**
 * @brief Get projection size by UID from network graph edges
 *
 * @param edges vector of network graph edges
 * @param edge_uid UID of the edge
 *
 * @return string size of the edge
 */
std::string get_proj_size(std::vector<knp::framework::NetworkGraph::Edge> const& edges, const knp::core::UID& edge_uid)
{
    auto it = std::find_if(
        edges.begin(), edges.end(), [&edge_uid](const auto& graph_edge) { return graph_edge.uid_ == edge_uid; });
    if (it != edges.end())
    {
        return std::to_string(it->size_);
    }
    throw std::runtime_error("Edge not found");
}
