/**
 * @file visualization.h
 * @brief Network graph building for the example.
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

#include <knp/framework/visualizer/visualize_network.h>

#include <cstddef>
#include <utility>

#include "annotated_network.h"


/**
 * @brief Collect data for a dynamic graph.
 *
 * @param network annotated network.
 * @param bus connections taken from the backend message bus.
 *
 * @return data for a dynamic graph.
 *
 * @details A handler and an output channel subscribe to their senders through their own message endpoints, so these
 * subscriptions are not registered in the backend and the senders are listed here. An input channel is visible on
 * the bus as a sender, only its kind is unknown there. An annotation field that is left empty makes the
 * corresponding node an external one, nothing else changes.
 */
inline knp::framework::DynamicInfo make_dynamic_info(
    const AnnotatedNetwork& network, knp::framework::BusConnections bus)
{
    knp::framework::DynamicInfo info;
    info.bus_ = std::move(bus);

    for (size_t i = 0; i < network.data_.wta_uids_.size(); ++i)
    {
        knp::framework::ExternalEntity handler;
        handler.uid_ = network.data_.wta_uids_[i];
        handler.name_ = "WTA";
        if (i < network.data_.wta_data_.size()) handler.senders_ = network.data_.wta_data_[i].first;
        info.entities_.push_back(handler);
    }

    // Channel names come from the annotation, where they are written next to the UIDs, so nothing is invented here.
    // An input channel sends spikes itself, so it has no senders of its own.
    for (const auto& [uid, name] : network.data_.input_channel_uids_)
    {
        info.entities_.push_back(knp::framework::ExternalEntity{uid, name, {}});
    }

    // An output channel listens to the populations it was connected to.
    for (const auto& [uid, name] : network.data_.output_channel_uids_)
    {
        info.entities_.push_back(knp::framework::ExternalEntity{uid, name, network.data_.output_uids_});
    }

    return info;
}


/**
 * @brief Build the static graph of an annotated network.
 *
 * @param network annotated network.
 *
 * @return network graph.
 *
 * @details The static graph shows what the network itself describes: populations, projections, names and sizes.
 */
inline knp::framework::NetworkGraph build_graph(const AnnotatedNetwork& network)
{
    return knp::framework::build_graph(network.network_);
}


/**
 * @brief Build the dynamic graph of an annotated network.
 *
 * @param network annotated network.
 * @param bus connections taken from the backend message bus.
 *
 * @return network graph.
 *
 * @details Connections come from the bus subscriptions. What the bus does not have, that is WTA handlers and
 * channels, comes from the annotation of the network.
 */
inline knp::framework::NetworkGraph build_graph(const AnnotatedNetwork& network, knp::framework::BusConnections bus)
{
    return knp::framework::build_graph(network.network_, make_dynamic_info(network, std::move(bus)));
}
