/**
 * @file visualization.h
 * @brief Functions for graph visualization.
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

#include <knp/framework/visualizer/graph_builder.h>

#include "annotated_network.h"


/**
 * @brief Assemble dynamic graph information from the annotated network and the backend message bus.
 *
 * @param network annotated network that contains static topology and additional annotation data, for example, WTA handlers, 
 * channel UIDs.
 * @param bus bus connections extracted from the training backend.
 *
 * @return `knp::framework::visualizer::DynamicInfo` object that merges runtime bus subscriptions with annotation level 
 * entities. This object produce a dynamic visualization that shows both static network structure and runtime 
 * communication links.
 *
 * @details A handler and an output channel subscribe to their senders through their own message endpoints, so these
 * subscriptions are not registered in the backend and the senders are listed here. An input channel is visible on
 * the bus as a sender, only its kind is unknown there. An annotation field that is left empty makes the
 * corresponding node an external one, nothing else changes.
 */
inline knp::framework::visualizer::DynamicInfo make_dynamic_info(
    const AnnotatedNetwork& network, const knp::framework::visualizer::BusConnections& bus)
{
    knp::framework::visualizer::DynamicInfo info;
    info.bus_ = bus;

    // Add WTA handlers, that is, handler UID and list of its senders.
    for (const auto& [uid, senders] : network.data_.wta_handlers_)
    {
        info.entities_.push_back(knp::framework::visualizer::ExternalEntity{uid, "WTA", senders});
    }

    // Add input channels. An input channel sends spikes itself, so it has no senders of its own.
    for (const auto& [uid, name] : network.data_.input_channel_uids_)
    {
        info.entities_.push_back(knp::framework::visualizer::ExternalEntity{uid, name, {}});
    }

    // Add output channels. An output channel listens to the populations it is connected to.
    for (const auto& [uid, name] : network.data_.output_channel_uids_)
    {
        info.entities_.push_back(knp::framework::visualizer::ExternalEntity{uid, name, network.data_.output_uids_});
    }

    return info;
}


/**
 * @brief Build a static network graph.
 *
 * @param network annotated network.
 *
 * @return network graph representing the static view of the network. No runtime bus information is included.
 *
 * @details The static graph is useful for inspecting the design of the network before any simulation is run.
 */
inline knp::framework::visualizer::NetworkGraph build_graph(const AnnotatedNetwork& network)
{
    return knp::framework::visualizer::build_graph(network.network_);
}


/**
 * @brief Build a dynamic network graph that combines static topology with runtime communication data.
 *
 * @param network annotated network.
 * @param bus bus connections obtained from the backend after training.
 *
 * @return network graph that visualizes both the static network elements and the dynamic entities, that is, WTA handlers,
 * input and output channels, and bus-based subscriptions.
 *
 * @details Connections come from message bus subscriptions. What the message bus does not have, for example, WTA handlers and
 * channels, comes from the annotation of the network.
 */
inline knp::framework::visualizer::NetworkGraph build_graph(
    const AnnotatedNetwork& network, const knp::framework::visualizer::BusConnections& bus)
{
    return knp::framework::visualizer::build_graph(network.network_, make_dynamic_info(network, bus));
}
