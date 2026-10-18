/**
 * @file visualize_network.h
 * @brief Network graph building, description and visualization.
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
#include <knp/framework/visualizer/network_graph.h>

#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Copy subscriptions of a message endpoint.
 *
 * @param endpoint message endpoint of a backend.
 *
 * @return connections that do not depend on the backend lifetime.
 *
 * @details Subscriptions are the wiring registered in the backend, not the message traffic. They are created when a
 * model is loaded and live as long as the backend does, so they can be copied at any moment after the loading.
 */
KNP_DECLSPEC BusConnections get_bus_connections(knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build network graph from a network (static structure).
 *
 * @param network source network for a graph.
 *
 * @return network graph.
 *
 * @details Populations are added as nodes and projections as edges. A projection with an unknown presynaptic or
 * postsynaptic UID is skipped, so the graph contains known static connections only.
 * Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=235801
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network);


/**
 * @brief Build network graph from a model (static structure).
 *
 * @param model source model for a graph.
 *
 * @return network graph.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model);


/**
 * @brief Build network graph from a network and message bus subscriptions (static and dynamic structure).
 *
 * @param network source network for a graph.
 * @param endpoint message endpoint of a backend.
 *
 * @return network graph.
 *
 * @details Connections are taken from the subscriptions only, because a loaded model rebuilds its projections and the
 * presynaptic and postsynaptic UIDs of the network do not match the real connections. The network gives names, sizes
 * and types. A spike message gives a projection source, a synaptic impact message gives a projection destination.
 * Entities that are not described by the network are added as nodes of kind "External". A projection that has no
 * subscriptions at all is skipped.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network, knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build network graph from a model and message bus subscriptions (static and dynamic structure).
 *
 * @param model source model for a graph.
 * @param endpoint message endpoint of a backend.
 *
 * @return network graph.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model, knp::core::MessageEndpoint& endpoint);


/**
 * @brief Build network graph from a network, message bus subscriptions and entities known to the calling code.
 *
 * @param network source network for a graph.
 * @param info endpoint of a backend and entities that cannot be restored from the bus.
 *
 * @return network graph.
 *
 * @details Works as the overload with an endpoint. In addition, an entity from @p info gets its own node kind, and
 * the senders listed for it are drawn as incoming edges. An entity that is not listed is drawn as "External".
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Network& network, const DynamicInfo& info);


/**
 * @brief Build network graph from a model, message bus subscriptions and entities known to the calling code.
 *
 * @param model source model for a graph.
 * @param info endpoint of a backend and entities that cannot be restored from the bus.
 *
 * @return network graph.
 */
KNP_DECLSPEC NetworkGraph build_graph(const knp::framework::Model& model, const DynamicInfo& info);


/**
 * @brief Print node and edge connections of a network graph.
 *
 * @param graph network graph.
 * @param out output stream, standard output by default.
 *
 * @details The function writes a line for each node and its incoming and outgoing edges. The same nodes and edges are
 * written as in the DOT text, so a description and a picture always match.
 */
KNP_DECLSPEC void print_network_description(const NetworkGraph& graph, std::ostream& out = std::cout);


/**
 * @brief Write network graph to a stream in DOT format.
 *
 * @param graph network graph.
 * @param out output stream.
 */
KNP_DECLSPEC void write_dot(const NetworkGraph& graph, std::ostream& out);


/**
 * @brief Render network graph to a stream as an image.
 *
 * @param graph network graph.
 * @param out output stream.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @return boolean indicating success.
 *
 * @details No intermediate file is created: the DOT text and the rendered image are kept in memory.
 */
KNP_DECLSPEC bool render_graph(const NetworkGraph& graph, std::ostream& out, const std::string& format = "png");


/**
 * @brief Save DOT and image files of a network graph.
 *
 * @param graph network graph.
 * @param name file name without extension.
 * @param dir directory for saving, it is created if it does not exist.
 * @param format graphviz output format ("png", "svg", "pdf" and so on).
 *
 * @return path to the image file.
 *
 * @throw std::filesystem::filesystem_error if a directory cannot be created.
 * @throw std::runtime_error if the DOT file cannot be written or the graph cannot be rendered.
 *
 * @details Files are saved as `<dir>/dot_files/<name>.dot` and `<dir>/<format>_files/<name>.<format>`. The DOT file is
 * written before rendering, so it is kept if rendering fails.
 */
KNP_DECLSPEC std::filesystem::path visualize(
    const NetworkGraph& graph, const std::string& name = "graph",
    const std::filesystem::path& dir = "visualization_docs", const std::string& format = "png");

}  // namespace knp::framework::visualizer
