/**
 * @file normal.h
 * @kaspersky_support Postnikov D.
 * @date 18.12.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include <knp/backends/cpu-library/impl/projections/delta/shared.h>

#include <vector>

#include "interface_fwd.h"

namespace knp::backends::cpu::projections::impl::delta::stdp
{
template <>
inline void init_synapse<AdditiveSTDPDeltaSynapse>(
    knp::synapse_traits::synapse_parameters<AdditiveSTDPDeltaSynapse> &params, uint64_t step)
{
}

class STDPFormula
{
public:
    STDPFormula(float tau_plus, float tau_minus, float a_plus, float a_minus)
        : tau_plus_(tau_plus), tau_minus_(tau_minus), a_plus_(a_plus), a_minus_(a_minus)
    {
    }

    [[nodiscard]] float stdp_w(float weight_diff) const
    {
        // Zhang et al. 1998.
        return weight_diff > 0 ? a_plus_ * std::exp(-weight_diff / tau_plus_)
                               : a_minus_ * std::exp(weight_diff / tau_minus_);
    }

    [[nodiscard]] float stdp_delta_w(
        const std::vector<knp::core::Step> &presynaptic_spikes,
        const std::vector<knp::core::Step> &postsynaptic_spikes) const
    {
        // Gerstner and al. 1996, Kempter et al. 1999.

        assert(presynaptic_spikes.size() == postsynaptic_spikes.size());

        float w_j = 0;

        for (const auto &t_f : presynaptic_spikes)
        {
            for (const auto &t_n : postsynaptic_spikes)
            {
                // cppcheck-suppress useStlAlgorithm
                w_j += stdp_w(static_cast<float>(t_n - t_f));
            }
        }
        return w_j;
    }

    [[nodiscard]] float operator()(
        const std::vector<knp::core::Step> &presynaptic_spikes,
        const std::vector<knp::core::Step> &postsynaptic_spikes) const
    {
        return stdp_delta_w(presynaptic_spikes, postsynaptic_spikes);
    }

private:
    float tau_plus_;
    float tau_minus_;
    float a_plus_;
    float a_minus_;
};

template <class DeltaLikeSynapse>
inline void append_spike_times(
    knp::core::Projection<AdditiveSTDPDeltaSynapse> &projection, const knp::core::messaging::SpikeMessage &message,
    const std::function<std::vector<size_t>(knp::core::messaging::SpikeIndex)> &synapse_index_getter,
    std::vector<knp::core::Step> knp::synapse_traits::STDPAdditiveRule<DeltaLikeSynapse>::*spike_queue)
{
    // Fill synapses spike queue.
    for (auto neuron_index : message.neuron_indexes_)
    {
        // Might be able to change it into "traces".
        // TODO: Inefficient, MUST be cached.
        for (auto synapse_index : synapse_index_getter(neuron_index))
        {
            auto &rule = std::get<core::SynapseElementAccess::synapse_data>(projection[synapse_index]).rule_;
            // Limit spike times queue.
            if ((rule.*spike_queue).size() < rule.tau_minus_ + rule.tau_plus_)
            {
                (rule.*spike_queue).push_back(message.header_.send_time_);
            }
        }
    }
}

template <>
inline void init_projection<AdditiveSTDPDeltaSynapse>(
    knp::core::Projection<AdditiveSTDPDeltaSynapse> &projection, std::vector<core::messaging::SpikeMessage> &messages,
    uint64_t step)
{
    using ProjectionType = typename std::decay_t<decltype(projection)>;
    using ProcessingType = typename ProjectionType::SharedSynapseParameters::ProcessingType;

    const auto &stdp_pops = projection.get_shared_parameters().stdp_populations_;

    // Spike messages to process as usual.
    std::vector<knp::core::messaging::SpikeMessage> stdp_only_messages;
    stdp_only_messages.reserve(messages.size());

    // TODO: Remove cycles.
    for (auto &msg : messages)
    {
        const auto &stdp_pop_iter = stdp_pops.find(msg.header_.sender_uid_);
        if (stdp_pop_iter == stdp_pops.end())
        {
            continue;
        }

        const auto &[uid, processing_type] = *stdp_pop_iter;
        assert(uid == msg.header_.sender_uid_);
        if (processing_type == ProcessingType::STDPOnly || processing_type == ProcessingType::STDPAndSpike)
        {
            append_spike_times(
                projection, msg,
                [&projection](uint32_t neuron_index)
                { return projection.find_synapses(neuron_index, ProjectionType::Search::by_postsynaptic); },
                &knp::synapse_traits::STDPAdditiveRule<knp::synapse_traits::DeltaSynapse>::postsynaptic_spike_times_);
        }
        if (processing_type == ProcessingType::STDPAndSpike)
        {
            append_spike_times(
                projection, msg,
                [&projection](uint32_t neuron_index)
                { return projection.find_synapses(neuron_index, ProjectionType::Search::by_postsynaptic); },
                &knp::synapse_traits::STDPAdditiveRule<knp::synapse_traits::DeltaSynapse>::presynaptic_spike_times_);
        }
        if (processing_type == ProcessingType::STDPOnly)
        {
            msg.neuron_indexes_ = {};
        }

        assert(processing_type == ProcessingType::STDPAndSpike || processing_type == ProcessingType::STDPOnly);
    }
}

template <>
inline void modify_weights<AdditiveSTDPDeltaSynapse>(knp::core::Projection<AdditiveSTDPDeltaSynapse> &projection)
{
    // Update projection parameters.
    for (uint64_t i = 0; i < projection.size(); ++i)
    {
        auto &proj = projection[i];
        auto &rule = std::get<knp::core::synapse_data>(proj).rule_;
        const auto period = rule.tau_plus_ + rule.tau_minus_;

        if (rule.presynaptic_spike_times_.size() >= period && rule.postsynaptic_spike_times_.size() >= period)
        {
            STDPFormula stdp_formula(rule.tau_plus_, rule.tau_minus_, 1, 1);
            std::get<knp::core::synapse_data>(proj).weight_ +=
                stdp_formula(rule.presynaptic_spike_times_, rule.postsynaptic_spike_times_);
            rule.presynaptic_spike_times_.clear();
            rule.postsynaptic_spike_times_.clear();
        }
    }
}

template <>
constexpr bool is_forced<AdditiveSTDPDeltaSynapse>()
{
    return false;
}

}  //namespace knp::backends::cpu::projections::impl::delta::stdp
