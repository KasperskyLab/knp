/**
 * @file delta_synapse_projection_impl.h
 * @brief Definition of DeltaSynapse calculation routines.
 * @kaspersky_support Artiom N.
 * @date 21.02.2023
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <knp/core/message_bus.h>
#include <knp/core/projection.h>
#include <knp/synapse-traits/delta.h>
#include <knp/synapse-traits/stdp_synaptic_resource_rule.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "additive_stdp_impl.h"


/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
/**
 * @brief Type of the message queue.
 */
using MessageQueue = std::unordered_map<uint64_t, knp::core::messaging::SynapticImpactMessage>;


template <class ProjectionType>
constexpr bool is_forcing()
{
    return false;
}


template <>
constexpr bool is_forcing<knp::core::Projection<synapse_traits::DeltaSynapse>>()
{
    return true;
}


template <typename ProjectionType>
MessageQueue::const_iterator calculate_delta_synapse_projection_data(
    ProjectionType &projection, std::vector<core::messaging::SpikeMessage> &messages, MessageQueue &future_messages,
    const size_t step_n,
    std::function<knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>(
        const typename ProjectionType::SynapseParameters &)>
        sp_getter = [](const typename ProjectionType::SynapseParameters &synapse_params) { return synapse_params; })
{
    SPDLOG_TRACE("Calculating delta synapse projection data for the step = {}", step_n);
    using SynapseType = typename ProjectionType::ProjectionSynapseType;
    WeightUpdateSTDP<SynapseType>::init_projection(projection, messages, step_n);

    for (const auto &message : messages)
    {
        const auto &message_data = message.neuron_indexes_;
        for (const auto &spiked_neuron_index : message_data)
        {
            auto synapses = projection.find_synapses(spiked_neuron_index, ProjectionType::Search::by_presynaptic);
            SPDLOG_TRACE("Projection synapse count for the spike = {}", synapses.size());
            for (auto synapse_index : synapses)
            {
                auto &synapse = projection[synapse_index];
                WeightUpdateSTDP<SynapseType>::init_synapse(std::get<core::synapse_data>(synapse), step_n);
                const auto &synapse_params = sp_getter(std::get<core::synapse_data>(synapse));

                // The message is sent on step N - 1, received on step N.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                knp::core::messaging::SynapticImpact impact{
                    synapse_index, synapse_params.weight_, synapse_params.output_type_,
                    static_cast<uint32_t>(std::get<core::source_neuron_id>(synapse)),
                    static_cast<uint32_t>(std::get<core::target_neuron_id>(synapse))};

                auto iter = future_messages.find(future_step);

                SPDLOG_TRACE("Synapse index = {}, synapse delay = {}, synapse weight = {}, step = {}, future step = {}",
                    synapse_index, synapse_params.delay_, synapse_params.weight_, step_n, future_step);

                if (iter != future_messages.end())
                {
                    SPDLOG_TRACE("Add existing impact.");
                    iter->second.impacts_.push_back(impact);
                }
                else
                {
                    knp::core::messaging::SynapticImpactMessage message_out{
                        {projection.get_uid(), step_n},
                        projection.get_presynaptic(),
                        projection.get_postsynaptic(),
                        is_forcing<ProjectionType>(),
                        {impact}};
                    SPDLOG_TRACE("Add new impact.");
                    future_messages.insert(std::make_pair(future_step, message_out));
                }
            }
        }
    }

    WeightUpdateSTDP<SynapseType>::modify_weights(projection);
    return future_messages.find(step_n);
}


template <class DeltaLikeSynapse>
void calculate_projection_part_impl(
    knp::core::Projection<DeltaLikeSynapse> &projection,
    const std::unordered_map<knp::core::Step, size_t> &message_in_data, MessageQueue &future_messages, uint64_t step_n,
    uint64_t part_start, uint64_t part_size, std::mutex &mutex)
{
    size_t part_end = std::min(part_start + part_size, static_cast<uint64_t>(projection.size()));
    std::vector<std::pair<uint64_t, knp::core::messaging::SynapticImpact>> container;
    WeightUpdateStdpMp<DeltaLikeSynapse>::init_projection_part(projection, message_in_data, step_n);
    for (size_t synapse_index = part_start; synapse_index < part_end; ++synapse_index)
    {
        auto &synapse = projection[synapse_index];
        // update_step(synapse.params_, step_n);
        auto iter = message_in_data.find(std::get<core::source_neuron_id>(synapse));
        if (iter == message_in_data.end())
        {
            continue;
        }

        // Add new impact.
        // The message is sent on step N - 1, received on step N.
        uint64_t key = std::get<core::synapse_data>(synapse).delay_ + step_n - 1;
        WeightUpdateStdpMp<DeltaLikeSynapse>::init_synapse(std::get<core::synapse_data>(synapse), step_n);

        knp::core::messaging::SynapticImpact impact{
            synapse_index, std::get<core::synapse_data>(synapse).weight_ * iter->second,
            std::get<core::synapse_data>(synapse).output_type_,
            static_cast<uint32_t>(std::get<core::source_neuron_id>(synapse)),
            static_cast<uint32_t>(std::get<core::target_neuron_id>(synapse))};

        container.emplace_back(key, impact);
    }
    // Add impacts to future messages queue, it is a shared resource.
    const std::lock_guard lock_guard(mutex);
    const auto &projection_uid = projection.get_uid();
    const auto &presynaptic_uid = projection.get_presynaptic();
    const auto &postsynaptic_uid = projection.get_postsynaptic();

    for (auto value : container)
    {
        auto iter = future_messages.find(value.first);
        if (iter != future_messages.end())
        {
            iter->second.impacts_.push_back(value.second);
        }
        else
        {
            knp::core::messaging::SynapticImpactMessage message_out{
                {projection_uid, step_n},
                postsynaptic_uid,
                presynaptic_uid,
                is_forcing<core::Projection<DeltaLikeSynapse>>(),
                {value.second}};
            future_messages.insert(std::make_pair(value.first, message_out));
        }
    }
    WeightUpdateStdpMp<DeltaLikeSynapse>::modify_weights_part(projection);
}


/**
 * @brief Convert spike vector to unordered map.
 * @param message spike message.
 * @return unordered map of `{index : number of instances}`. Number of instances usually equals `1`.
 */
inline std::unordered_map<uint64_t, size_t> convert_spikes(const core::messaging::SpikeMessage &message)
{
    std::unordered_map<knp::core::Step, size_t> result;
    for (auto neuron_idx : message.neuron_indexes_)
    {
        auto iter = result.find(neuron_idx);
        if (result.end() == iter)
        {
            result.insert({neuron_idx, 1});
        }
        else
        {
            ++(iter->second);
        }
    }
    return result;
}


template <class DeltaLikeSynapseType>
void calculate_delta_synapse_projection_impl(
    knp::core::Projection<DeltaLikeSynapseType> &projection, knp::core::MessageEndpoint &endpoint,
    MessageQueue &future_messages, size_t step_n)
{
    SPDLOG_DEBUG("Calculating delta synapse projection...");

    auto messages = endpoint.unload_messages<core::messaging::SpikeMessage>(projection.get_uid());

    SPDLOG_DEBUG("Messages count = {}...", messages.size());

    #if (SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE)
    for (const auto &message : messages)
    {
            SPDLOG_TRACE("Spike from {} with spiked neurons: {}",
                std::string(message.header_.sender_uid_),
                fmt::join(message.neuron_indexes_, ", "));
    }
    #endif

    auto out_iter = calculate_delta_synapse_projection_data(projection, messages, future_messages, step_n);
    if (out_iter != future_messages.end())
    {
        SPDLOG_TRACE("Projection is sending an impact message.");
        // Send a message and remove it from the queue.
        endpoint.send_message(out_iter->second);
        future_messages.erase(out_iter);
    }
}

}  // namespace knp::backends::cpu
