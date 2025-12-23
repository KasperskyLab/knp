/**
 * @file additive_stdp_impl.h
 * @brief Implementation of additive STDP calculation routines.
 * @kaspersky_support Artiom N.
 * @date 21.08.2023
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
#include <knp/backends/cpu-library/impl/base_stdp_impl.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/messaging/synaptic_impact_message.h>
#include <knp/core/projection.h>
#include <knp/synapse-traits/stdp_common.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>


namespace knp::backends::cpu
{

using knp::core::UID;
using knp::core::messaging::SpikeMessage;


inline const knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse> &get_delta_synapse_params(
    const knp::synapse_traits::synapse_parameters<
        knp::synapse_traits::STDP<knp::synapse_traits::STDPAdditiveRule, knp::synapse_traits::DeltaSynapse>>
        &synapse_params)
{
    return synapse_params;
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
    knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection, const SpikeMessage &message,
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


inline void append_spike_times(
    knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
    const std::vector<SpikeMessage> &spikes, const std::function<std::vector<size_t>(uint32_t)> &syn_index_getter,
    std::vector<knp::core::Step> knp::synapse_traits::STDPAdditiveRule<knp::synapse_traits::DeltaSynapse>::*spike_queue)
{
    for (const auto &msg : spikes)
    {
        append_spike_times(projection, msg, syn_index_getter, spike_queue);
    }
}


template <class SynapseType>
constexpr bool is_additive_stdp_synapse()
{
    if constexpr (!std::is_same_v<
                      SynapseType, synapse_traits::STDP<synapse_traits::STDPAdditiveRule, typename SynapseType::Base>>)
    {
        return is_additive_stdp_synapse<typename SynapseType::Base>();
    }

    using AddSTDPProj =
        typename knp::core::Projection<synapse_traits::STDP<synapse_traits::STDPAdditiveRule, SynapseType>>;
    return std::is_same<std::decay_t<SynapseType>, AddSTDPProj>::value;
}


template <class DeltaLikeSynapse>
void register_additive_stdp_spikes_part(
    knp::core::Projection<knp::synapse_traits::STDP<knp::synapse_traits::STDPAdditiveRule, DeltaLikeSynapse>>
        &projection,
    std::vector<SpikeMessage> &all_messages, uint64_t part_start, uint64_t part_end)
{
    if (part_start != 0) return;  // Not much sense to parallelize this by projection, so it's calculated just once.
    SPDLOG_DEBUG("Calculating additive STDP delta synapse projection...");

    using ProjectionType = typename std::decay_t<decltype(projection)>;
    using ProcessingType = typename ProjectionType::SharedSynapseParameters::ProcessingType;

    const auto &stdp_pops = projection.get_shared_parameters().stdp_populations_;

    // Spike messages to process as usual.
    std::vector<SpikeMessage> stdp_only_messages;
    stdp_only_messages.reserve(all_messages.size());

    // TODO: Remove cycles.
    for (auto &msg : all_messages)
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
            SPDLOG_TRACE("Add spikes to STDP projection postsynaptic history.");
            append_spike_times(
                projection, msg,
                [&projection](uint32_t neuron_index)
                { return projection.find_synapses(neuron_index, ProjectionType::Search::by_postsynaptic); },
                &knp::synapse_traits::STDPAdditiveRule<knp::synapse_traits::DeltaSynapse>::postsynaptic_spike_times_);
        }
        if (processing_type == ProcessingType::STDPAndSpike)
        {
            SPDLOG_TRACE("Add spikes to STDP projection presynaptic history.");
            append_spike_times(
                projection, msg,
                [&projection](uint32_t neuron_index)
                { return projection.find_synapses(neuron_index, ProjectionType::Search::by_postsynaptic); },
                &knp::synapse_traits::STDPAdditiveRule<knp::synapse_traits::DeltaSynapse>::presynaptic_spike_times_);
        }
        if (processing_type == ProcessingType::STDPOnly)
        {
            SPDLOG_TRACE("STDP-only synapse, remove message from list.");
            msg.neuron_indexes_ = {};
        }

        assert(processing_type == ProcessingType::STDPAndSpike || processing_type == ProcessingType::STDPOnly);
    }
}


template <class DeltaLikeSynapse>
void update_projection_weights_additive_stdp_part(
    knp::core::Projection<knp::synapse_traits::STDP<knp::synapse_traits::STDPAdditiveRule, DeltaLikeSynapse>>
        &projection,
    uint64_t part_start, uint64_t part_end)
{
    // Update projection parameters.
    for (uint64_t i = part_start; i < std::min<uint64_t>(projection.size(), part_end); ++i)
    {
        auto &proj = projection[i];
        SPDLOG_TRACE("Applying STDP rule...");
        auto &rule = std::get<knp::core::synapse_data>(proj).rule_;
        const auto period = rule.tau_plus_ + rule.tau_minus_;

        if (rule.presynaptic_spike_times_.size() >= period && rule.postsynaptic_spike_times_.size() >= period)
        {
            STDPFormula stdp_formula(rule.tau_plus_, rule.tau_minus_, 1, 1);
            SPDLOG_TRACE("Old weight = {}.", std::get<knp::core::synapse_data>(proj).weight_);
            std::get<knp::core::synapse_data>(proj).weight_ +=
                stdp_formula(rule.presynaptic_spike_times_, rule.postsynaptic_spike_times_);
            SPDLOG_TRACE("New weight = {}.", std::get<knp::core::synapse_data>(proj).weight_);
            rule.presynaptic_spike_times_.clear();
            rule.postsynaptic_spike_times_.clear();
        }
    }
}


template <class DeltaLikeSynapse>
struct WeightUpdateSTDP<synapse_traits::STDP<synapse_traits::STDPAdditiveRule, DeltaLikeSynapse>>
{
    using Synapse = synapse_traits::STDP<synapse_traits::STDPAdditiveRule, DeltaLikeSynapse>;

    static void init_projection_part(
        knp::core::Projection<Synapse> &projection, std::vector<SpikeMessage> &all_messages, knp::core::Step step,
        uint64_t part_start, uint64_t part_end)
    {
        register_additive_stdp_spikes_part(projection, all_messages, part_start, part_end);
    }

    static void init_synapse(const knp::synapse_traits::synapse_parameters<Synapse> &projection, knp::core::Step step)
    {
    }

    static void modify_weights_part(knp::core::Projection<Synapse> &projection, uint64_t part_start, uint64_t part_end)
    {
        update_projection_weights_additive_stdp_part(projection, part_start, part_end);
    }

    static void init_projection(
        knp::core::Projection<Synapse> &projection, std::vector<SpikeMessage> &all_messages, knp::core::Step step)
    {
        init_projection_part(projection, all_messages, step, 0, projection.size());
    }

    static void modify_weights(knp::core::Projection<Synapse> &projection)
    {
        modify_weights_part(projection, 0, projection.size());
    }
};


}  // namespace knp::backends::cpu
