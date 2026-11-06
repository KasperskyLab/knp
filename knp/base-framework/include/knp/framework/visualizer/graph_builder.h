/**
 * @file graph_builder.h
 * @brief Network graph building.
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

#include <knp/core/message_endpoint.h>
#include <knp/framework/model.h>
#include <knp/framework/network.h>
#include <knp/framework/visualizer/dynamic_info.h>
#include <knp/framework/visualizer/network_graph.h>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{


/**
 * @brief Extract a copy of all subscriptions from a backend message endpoint.
 *
 * @param endpoint reference to the backend's `MessageEndpoint`.

 * @return `BusConnections` object that contains only the subscription pairs.
 *
 * @details Subscriptions are the wiring registered in the backend, not the message traffic. They are created when a
 * model is loaded and live as long as the backend does, so they can be copied at any moment after the loading.
 * The returned data is independent of the backend lifetime; it can be stored,
 * transferred across threads, or used after the backend has been destroyed.
 */
KNP_DECLSPEC BusConnections get_bus_connections(knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build a static graph from a `Network`.
 *
 * @details Populations become nodes, projections become edges. Projections whose presynaptic
 * or postsynaptic UID is unknown are omitted, so the resulting graph contains only
 * fully described static connections.
 *
 * @param network source `Network` object.
 *
 * @return a fully constructed `NetworkGraph`.
 *
 * @note Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network);


/**
 * @brief Build a static graph from a `Model`.
 *
 * @details The model's underlying network is extracted and then processed as in the overload
 * that takes a `Network`.
 *
 * @param model source `Model` object.
 *
 * @return a `NetworkGraph` representing the static topology.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model);


/**
 * @brief Build a dynamic graph from a `Network` and a backend endpoint.
 *
 * @details The static network supplies node names, sizes and types, while the bus subscriptions
 * provide the actual connections. Entities that are not described
 * by the network are rendered with the node kind "External".
 * Projections without any subscription are omitted. Spike messages give the projection as the receiver, 
 * synaptic impact messages give the projection as the sender.
 *
 * @param network static `Network`.
 * @param endpoint backend's `MessageEndpoint` providing bus subscriptions.
 *
 * @return a `NetworkGraph` that mixes static information with dynamic bus-driven edges.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network, knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build a dynamic graph from a `Model` and a backend endpoint.
 *
 * @details Equivalent to the overload that takes a `Network`; the model's network is extracted
 * automatically.
 *
 * @param model source `Model`.
 * @param endpoint backend's `MessageEndpoint`.
 *
 * @return a `NetworkGraph` containing static and dynamic information.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model, knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build a dynamic graph from a `Network` together with explicit external entities.
 *
 * @details In addition to the bus subscriptions, the caller can provide a list of entities, for example, 
 * handlers, and channels, in @p info that are not recoverable from the bus. Each such entity
 * receives its own node kind as defined in `DynamicInfo::entities_`.
 *
 * @param network static `Network`.
 * @param info combined bus connections and external entities.
 *
 * @return a `NetworkGraph` that reflects both static topology and the supplied dynamic
 * information.
 *
 * @note This overload is the most flexible: it can be used when the caller already
 * knows the `DynamicInfo`, for example, after a custom preprocessing step.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network, const DynamicInfo& info);


/**
 * @brief Build a dynamic graph from a `Model` together with explicit external entities.
 *
 * @details Same semantics as the previous overload, but the network is taken from the model.
 *
 * @param model source `Model`.
 * @param info combined bus connections and external entities.
 *
 * @return a `NetworkGraph` that mixes static network data with the supplied dynamic data.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model, const DynamicInfo& info);

}  // namespace knp::framework::visualizer
