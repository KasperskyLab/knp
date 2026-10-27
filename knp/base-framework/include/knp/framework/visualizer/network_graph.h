/**
 * @file network_graph.h
 * @brief Network graph built from a network or from message bus subscriptions.
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

#include <knp/core/impexp.h>

#include <memory>


/**
 * @brief Implementation namespace of the network visualizer.
 */
namespace knp::framework::visualizer::impl
{
/**
 * @brief The NetworkGraphImpl class is an internal implementation class for network graph.
 */
class NetworkGraphImpl;
}  // namespace knp::framework::visualizer::impl


/**
 * @brief Network visualizer namespace.
 */
namespace knp::framework::visualizer
{

/**
 * @brief Network graph built from a network or from message bus subscriptions.
 *
 * @details The graph is built by `build_graph` and passed to the description and visualization functions. Its
 * nodes and edges are internal.
 */
class KNP_DECLSPEC NetworkGraph
{
public:
    /**
     * @brief Network graph constructor with an implementation.
     *
     * @param impl network graph implementation.
     */
    explicit NetworkGraph(std::unique_ptr<impl::NetworkGraphImpl> &&impl);

    /**
     * @brief Move constructor.
     */
    NetworkGraph(NetworkGraph &&) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * @return reference to the graph.
     */
    NetworkGraph &operator=(NetworkGraph &&) noexcept;

    /**
     * @brief Destructor.
     */
    ~NetworkGraph();

    /**
     * @brief Check that the graph was built with message bus subscriptions.
     *
     * @return `true` for a dynamic graph.
     */
    [[nodiscard]] bool is_dynamic() const;

    /**
     * @brief Get network graph implementation.
     *
     * @return network graph implementation.
     */
    [[nodiscard]] const impl::NetworkGraphImpl &get_impl() const;

private:
    /**
     * @brief Network graph implementation.
     */
    std::unique_ptr<impl::NetworkGraphImpl> impl_;
};

}  // namespace knp::framework::visualizer
