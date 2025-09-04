/**
 * @file synaptic_resource_stdp_impl.h
 * @brief Synaptic resource based STDP neuron adapter.
 * @kaspersky_support Artiom N.
 * @date 06.10.2023
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
#include <knp/backends/cpu-library/impl/base_stdp_impl.h>
#include <knp/core/messaging/spike_message.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/core/uid.h>
#include <knp/synapse-traits/stdp_common.h>
#include <knp/synapse-traits/stdp_synaptic_resource_rule.h>
#include <knp/synapse-traits/stdp_type_traits.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/mp11.hpp>


/**
 * @brief Namespace for neuron traits.
 */
namespace knp::backends::cpu
{


template <class SynapseType, class ProjectionContainer>
std::vector<knp::core::Projection<SynapseType> *> find_projection_by_type_and_postsynaptic(
    ProjectionContainer &projections, const knp::core::UID &post_uid, bool exclude_locked)
{
    using ProjectionType = knp::core::Projection<SynapseType>;
    std::vector<knp::core::Projection<SynapseType> *> result;
    constexpr auto type_index = boost::mp11::mp_find<synapse_traits::AllSynapses, SynapseType>();
    for (auto &projection : projections)
    {
        if (projection.arg_.index() != type_index)
        {
            continue;
        }

        ProjectionType *projection_ptr = &(std::get<type_index>(projection.arg_));
        if (projection_ptr->is_locked() && exclude_locked)
        {
            continue;
        }

        if (projection_ptr->get_postsynaptic() == post_uid)
        {
            result.push_back(projection_ptr);
        }
    }
    return result;
}


template <class Synapse>
using STDPSynapseParams = knp::synapse_traits::synapse_parameters<
    knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, Synapse>>;


/**
 * @brief Recalculate synapse weights from synaptic resource.
 * @tparam WeightedSynapse synapse that has `weight_` parameter.
 * @param synapse_params synapse parameters.
 */
template <class WeightedSynapse>
void recalculate_synapse_weights(std::vector<STDPSynapseParams<WeightedSynapse> *> &synapse_params)
{
    // Synapse weight recalculation.
    for (auto synapse_ptr : synapse_params)
    {
        const auto syn_w = std::max(synapse_ptr->rule_.synaptic_resource_, 0.F);
        const auto weight_diff = synapse_ptr->rule_.w_max_ - synapse_ptr->rule_.w_min_;
        synapse_ptr->weight_ = synapse_ptr->rule_.w_min_ + weight_diff * syn_w / (weight_diff + syn_w);
    }
}


template <class Synapse>
using StdpProjection =
    knp::core::Projection<knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, Synapse>>;


inline bool is_point_in_interval(uint64_t interval_begin, uint64_t interval_end, uint64_t point)
{
    const bool is_after_begin = point >= interval_begin;
    const bool is_before_end = point <= interval_end;
    const bool is_overflow = interval_end < interval_begin;
    return (is_after_begin && is_before_end) || ((is_after_begin || is_before_end) && is_overflow);
}


template <class SynapseType>
std::vector<synapse_traits::synapse_parameters<SynapseType> *> get_all_connected_synapses(
    const std::vector<core::Projection<SynapseType> *> &projections_to_neuron, size_t neuron_index)
{
    std::vector<synapse_traits::synapse_parameters<SynapseType> *> result;
    for (auto *projection : projections_to_neuron)
    {
        auto synapses = projection->find_synapses(neuron_index, core::Projection<SynapseType>::Search::by_postsynaptic);
        std::transform(
            synapses.begin(), synapses.end(), std::back_inserter(result),
            [&projection](auto const &index) { return &std::get<core::synapse_data>((*projection)[index]); });
    }
    return result;
}


/**
 * @brief Update spike sequence state for the neuron. It's called after a neuron sends a spike.
 * @tparam NeuronType base neuron type.
 * @param neuron neuron parameters.
 * @param step current step.
 * @return new state.
 */
template <class NeuronType>
neuron_traits::ISIPeriodType update_isi(
    neuron_traits::neuron_parameters<neuron_traits::SynapticResourceSTDPNeuron<NeuronType>> &neuron, uint64_t step)
{
    // This neuron got a forcing spike this turn and doesn't continue its spiking sequence.
    if (neuron.is_being_forced_)
    {
        neuron.isi_status_ = neuron_traits::ISIPeriodType::is_forced;
        // Do not update last_step_.
        return neuron.isi_status_;
    }

    switch (neuron.isi_status_)
    {
        case neuron_traits::ISIPeriodType::not_in_period:
        case neuron_traits::ISIPeriodType::is_forced:
            neuron.isi_status_ = neuron_traits::ISIPeriodType::period_started;
            neuron.first_isi_spike_ = step;
            break;
        case neuron_traits::ISIPeriodType::period_started:
            if (neuron.last_step_ - step < neuron.isi_max_)
            {
                neuron.isi_status_ = neuron_traits::ISIPeriodType::period_continued;
            }
            break;
        case neuron_traits::ISIPeriodType::period_continued:
            if (neuron.last_step_ - step >= neuron.isi_max_ || neuron.dopamine_value_ != 0)
            {
                neuron.isi_status_ = neuron_traits::ISIPeriodType::period_started;
                neuron.first_isi_spike_ = step;
            }
            break;
        default:
            throw std::runtime_error("Not supported ISI status.");
    }

    neuron.last_step_ = step;
    return neuron.isi_status_;
}


/**
 * @brief Apply STDP to all presynaptic connections of a single population.
 * @tparam NeuronType type of neuron that is compatible with STDP.
 * @param msg spikes emited by population.
 * @param working_projections all projections (those that are not connected, locked or are of a wrong type are
 * skipped).
 * @param population population.
 * @param step current network step.
 * @note all projections are supposed to be of the same type.
 */
template <class NeuronType>
void process_spiking_neurons(
    const core::messaging::SpikeMessage &msg,
    std::vector<StdpProjection<synapse_traits::DeltaSynapse> *> &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPNeuron<NeuronType>> &population, uint64_t step);


/**
 * @brief If a neuron resource is greater than `1` or `-1` it should be distributed among all synapses.
 * @tparam NeuronType type of base neuron (BLIFAT for SynapticResourceSTDPBlifat).
 * @param working_projections list of STDP projections (`DeltaSynapse` only is supported now).
 * @param population reference to population.
 * @param step current step.
 */
template <class NeuronType>
void renormalize_resource(
    std::vector<StdpProjection<synapse_traits::DeltaSynapse> *> &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPNeuron<NeuronType>> &population, uint64_t step)
{
    using SynapseType =
        knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, synapse_traits::DeltaSynapse>;
    for (size_t neuron_index = 0; neuron_index < population.size(); ++neuron_index)
    {
        auto &neuron = population[neuron_index];
        if (step - neuron.last_step_ <= neuron.isi_max_ &&
            neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced)
        {
            // Neuron is still in ISI period, skip it.
            continue;
        }

        if (std::fabs(neuron.free_synaptic_resource_) < neuron.synaptic_resource_threshold_)
        {
            continue;
        }

        auto synapse_params = get_all_connected_synapses<SynapseType>(working_projections, neuron_index);

        // Divide free resource between all synapses.
        auto add_resource_value =
            neuron.free_synaptic_resource_ / (synapse_params.size() + neuron.resource_drain_coefficient_);

        for (auto *synapse : synapse_params)
        {
            synapse->rule_.synaptic_resource_ += add_resource_value;
        }

        neuron.free_synaptic_resource_ = 0.0F;
        recalculate_synapse_weights(synapse_params);
    }
}


template <class NeuronType>
void do_dopamine_plasticity(
    std::vector<StdpProjection<synapse_traits::DeltaSynapse> *> &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPNeuron<NeuronType>> &population, uint64_t step);


template <class DeltaLikeSynapse>
struct WeightUpdateSTDP<synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, DeltaLikeSynapse>>
{
    using Synapse = synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, DeltaLikeSynapse>;
    static void init_projection(
        const knp::core::Projection<Synapse> &projection, const std::vector<core::messaging::SpikeMessage> &messages,
        uint64_t step)
    {
    }

    static void init_synapse(knp::synapse_traits::synapse_parameters<Synapse> &params, uint64_t step)
    {
        params.rule_.last_spike_step_ = step;
    }

    static void modify_weights(const knp::core::Projection<Synapse> &projection) {}
};


template <class DeltaLikeSynapse>
struct WeightUpdateStdpMp
{
    using Synapse = DeltaLikeSynapse;
    static void init_projection_part(
        const knp::core::Projection<Synapse> &projection,
        const std::unordered_map<knp::core::Step, size_t> &message_data, uint64_t step)
    {
    }

    static void init_synapse(knp::synapse_traits::synapse_parameters<Synapse> &params, uint64_t step) {}

    static void modify_weights_part(const knp::core::Projection<Synapse> &projection) {}
};


template <class DeltaLikeSynapse>
struct WeightUpdateStdpMp<synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, DeltaLikeSynapse>>
{
    using Synapse = synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, DeltaLikeSynapse>;
    static void init_projection_part(
        const knp::core::Projection<Synapse> &projection,
        const std::unordered_map<knp::core::Step, size_t> &message_data, uint64_t step)
    {
    }

    static void init_synapse(knp::synapse_traits::synapse_parameters<Synapse> &params, uint64_t step)
    {
        params.rule_.last_spike_step_ = step;
    }

    static void modify_weights_part(const knp::core::Projection<Synapse> &projection) {}
};

template <class NeuronType, class SynapseType>
void do_STDP_resource_plasticity(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPNeuron<NeuronType>> &population,
    std::vector<knp::core::Projection<synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, SynapseType>> *>
        working_projections,
    const std::optional<core::messaging::SpikeMessage> &message, uint64_t step)
{
    // Call learning functions on all found projections:
    // 1. If neurons generated spikes, process these neurons.
    if (message.has_value())
    {
        knp::backends::cpu::process_spiking_neurons<NeuronType>(message.value(), working_projections, population, step);
    }

    // 2. Do dopamine plasticity.
    knp::backends::cpu::do_dopamine_plasticity(working_projections, population, step);

    // 3. Renormalize resources if needed.
    knp::backends::cpu::renormalize_resource(working_projections, population, step);
}


}  // namespace knp::backends::cpu
