/**
 * @file impl.h
 * @kaspersky_support Postnikov D.
 * @date 10.12.2025
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

#include <knp/backends/cpu-library/temp_impl/projections/shared/def.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/projection.h>

#include <utility>
#include <vector>

#include "shared.h"

namespace knp::backends::cpu::projections::delta
{

template <typename DeltaLikeSynapse>
inline MessageQueue::const_iterator calculate_projection_impl(
    knp::core::Projection<DeltaLikeSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    MessageQueue &future_messages, size_t step_n)
{
    using ProjectionType = knp::core::Projection<DeltaLikeSynapse>;

    for (const auto &message : messages)
    {
        const auto &message_data = message.neuron_indexes_;
        for (const auto &spiked_neuron_index : message_data)
        {
            auto synapses = projection.find_synapses(spiked_neuron_index, ProjectionType::Search::by_presynaptic);
            for (auto synapse_index : synapses)
            {
                auto &synapse = projection[synapse_index];
                if constexpr (std::is_same_v<DeltaLikeSynapse, STDPDeltaSynapse>)
                {
                    std::get<core::synapse_data>(synapse).rule_.last_spike_step_ = step_n;
                }
                const auto &synapse_params = std::get<core::synapse_data>(synapse);

                // The message is sent on step N - 1, received on step N.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                knp::core::messaging::SynapticImpact impact{
                    synapse_index, synapse_params.weight_, synapse_params.output_type_,
                    static_cast<uint32_t>(std::get<core::source_neuron_id>(synapse)),
                    static_cast<uint32_t>(std::get<core::target_neuron_id>(synapse))};

                auto iter = future_messages.find(future_step);
                if (iter != future_messages.end())
                {
                    iter->second.impacts_.push_back(impact);
                }
                else
                {
                    bool is_forced = false;
                    if constexpr (std::is_same_v<DeltaLikeSynapse, DeltaSynapse>)
                    {
                        is_forced = true;
                    }
                    knp::core::messaging::SynapticImpactMessage message_out{
                        {projection.get_uid(), step_n},
                        projection.get_presynaptic(),
                        projection.get_postsynaptic(),
                        is_forced,
                        {impact}};
                    future_messages.insert(std::make_pair(future_step, message_out));
                }
            }
        }
    }
    return future_messages.find(step_n);
}


}  //namespace knp::backends::cpu::projections::delta
