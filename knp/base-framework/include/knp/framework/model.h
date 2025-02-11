/**
 * @file model.h
 * @brief Model interface.
 * @kaspersky_support Artiom N.
 * @date 30.03.2023
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

#include <knp/core/core.h>
#include <knp/core/impexp.h>
#include <knp/framework/backend_loader.h>
#include <knp/framework/io/input_channel.h>
#include <knp/framework/io/output_channel.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief The Model class is a definition of a model that contains a network, input and output channels, a monitor and a
 * backend.
 */
class KNP_DECLSPEC Model
{
public:
    /**
     * @brief Model constructor.
     * @param network network to use in the model.
     */
    explicit Model(knp::framework::Network &&network) : network_(std::move(network)) {}

public:
    /**
     * @brief Get model UID.
     * @return model UID.
     */
    [[nodiscard]] const knp::core::UID &get_uid() const { return base_.uid_; }
    /**
     * @brief Get tags used by the model.
     * @return model tag map.
     * @see TagMap.
     */
    [[nodiscard]] auto &get_tags() { return base_.tags_; }

public:
    /**
     * @brief Get network associated with the model.
     * @return network.
     */
    [[nodiscard]] knp::framework::Network &get_network() { return network_; }
    /**
     * @brief Get network associated with the model.
     * @note Constant method.
     * @return network.
     */
    [[nodiscard]] const knp::framework::Network &get_network() const { return network_; }

    /**
     * @brief Add an input channel to the network.
     * @param channel_uid UID of the input channel.
     * @param projection_uid UID of the projection which will be connected to the channel.
     */
    void add_input_channel(const core::UID &channel_uid, const core::UID &projection_uid);

    /**
     * @brief Add an input channel to the network.
     * @param channel_uid UID of the input channel.
     * @param projection projection which will be connected to the channel.
     */
    template <typename SynapseType>
    void connect_input_projection(const core::UID &channel_uid, const core::Projection<SynapseType> &projection)
    {
        add_input_channel(channel_uid, projection.get_uid());
    }

    /**
     * @brief Add an output channel to the network.
     * @param channel_uid UID of the channel object.
     * @param population_uid UID of the population which will be connected to the channel.
     */
    void add_output_channel(const core::UID &channel_uid, const core::UID &population_uid);

    /**
     * @brief Add an output channel to the network.
     * @param channel_uid UID of the channel object.
     * @param population population which will be connected to the channel.
     */
    template <typename NeuronType>
    void connect_output_population(const core::UID &channel_uid, const core::Population<NeuronType> &population)
    {
        add_output_channel(channel_uid, population.get_uid());
    }

    /**
     * @brief Return all input channels UIDs.
     * @return map of input channels to projections.
     */
    const std::unordered_multimap<core::UID, core::UID, core::uid_hash> &get_input_channels() const;
    /**
     * @brief Return all output channels UIDs.
     * @return map of output channels to populations.
     */
    const std::unordered_multimap<core::UID, core::UID, core::uid_hash> &get_output_channels() const;

private:
    knp::core::BaseData base_;
    knp::framework::Network network_;
    std::unordered_multimap<core::UID, core::UID, core::uid_hash> in_channels_;
    std::unordered_multimap<core::UID, core::UID, core::uid_hash> out_channels_;
};

}  // namespace knp::framework
