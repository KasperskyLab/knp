/**
 * @file delta_impl.h
 * @kaspersky_support Vartenkov A.
 * @date 10.12.2025
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

#include <knp/backends/cpu-library/impl/projections/message_queue.h>
#include <knp/backends/cpu-library/impl/projections/training/stdp/stdp_dispatcher.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/projection.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>


namespace knp::backends::cpu::projections::impl::delta
{

/**
 * @brief Delta synapse shortcut.
 */
using DeltaSynapse = knp::synapse_traits::DeltaSynapse;

/**
 * @brief STDP Delta synapse shortcut.
 */
using STDPDeltaSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;

/**
 * @brief Additive STDP Delta synapse shortcut.
 */
using AdditiveSTDPDeltaSynapse = knp::synapse_traits::AdditiveSTDPDeltaSynapse;


template <typename DeltaLikeSynapse>
MessageQueue::const_iterator calculate_projection_impl(
    knp::core::Projection<DeltaLikeSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    MessageQueue &future_messages, size_t step_n)
{
    using ProjectionType = knp::core::Projection<DeltaLikeSynapse>;

    SPDLOG_TRACE("Calculating delta synapse projection data for the step = {}", step_n);

    training::stdp::init_projection(projection, messages, step_n);

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
                training::stdp::init_synapse(std::get<core::synapse_data>(synapse), step_n);
                const auto &synapse_params = std::get<core::synapse_data>(synapse);

                // The message is sent on step N - 1, received on step N.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                knp::core::messaging::SynapticImpact impact{
                    synapse_index, synapse_params.weight_, synapse_params.output_type_,
                    static_cast<uint32_t>(std::get<core::source_neuron_id>(synapse)),
                    static_cast<uint32_t>(std::get<core::target_neuron_id>(synapse))};

                auto iter = future_messages.find(future_step);

                SPDLOG_TRACE(
                    "Synapse index = {}, synapse delay = {}, synapse weight = {}, step = {}, future step = {}",
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
                        training::stdp::is_forced(projection),
                        {impact}};
                    SPDLOG_TRACE("Add new impact.");
                    future_messages.insert(std::make_pair(future_step, message_out));
                }
            }
        }
    }

    training::stdp::modify_weights(projection);

    return future_messages.find(step_n);
}


template <class DeltaLikeSynapse>
void calculate_projection_multithreaded_impl(
    knp::core::Projection<DeltaLikeSynapse> &projection,
    const std::unordered_map<knp::core::Step, size_t> &message_in_data, MessageQueue &future_messages, uint64_t step_n,
    uint64_t part_start, uint64_t part_size, std::mutex &mutex)
{
    size_t part_end = std::min(part_start + part_size, static_cast<uint64_t>(projection.size()));
    std::vector<std::pair<uint64_t, knp::core::messaging::SynapticImpact>> container;
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
        if constexpr (std::is_same_v<DeltaLikeSynapse, STDPDeltaSynapse>)
        {
            std::get<core::synapse_data>(synapse).rule_.last_spike_step_ = step_n;
        }

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
                training::stdp::is_forced(projection),
                {value.second}};
            future_messages.insert(std::make_pair(value.first, message_out));
        }
    }
}

}  //namespace knp::backends::cpu::projections::impl::delta
