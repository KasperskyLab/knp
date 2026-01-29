/**
 * @file stdp.h
 * @brief Shared stdp related functions.
 * @kaspersky_support Artiom N.
 * @date 08.12.2025
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

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>


namespace knp::backends::cpu::populations::impl::training::stdp
{

/**
 * @brief Recalculate synapse weights from synaptic resource.
 * @tparam Synapse Base synapse type.
 * @param synapse_params Synapse parameters.
 */
template <class Synapse>
void recalculate_synapse_weights(
    std::vector<std::reference_wrapper<knp::synapse_traits::synapse_parameters<
        knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, Synapse>>>> &synapse_params)
{
    // Synapse weight recalculation.
    for (auto &synapse : synapse_params)
    {
        const auto &rule = synapse.get().rule_;
        const auto syn_w = std::max(rule.synaptic_resource_, 0.F);
        const auto weight_diff = rule.w_max_ - rule.w_min_;
        synapse.get().weight_ = rule.w_min_ + weight_diff * syn_w / (weight_diff + syn_w);
    }
}


/**
 * @brief Get all synapses that are connected to some neuron.
 * @tparam Synapse Synapse type.
 * @param projections All projections that lead to neuron's population.
 * @param neuron_index Neuron index.
 * @return Connected synapses.
 */
template <class Synapse>
std::vector<std::reference_wrapper<synapse_traits::synapse_parameters<Synapse>>> get_all_connected_synapses(
    std::vector<std::reference_wrapper<core::Projection<Synapse>>> &projections, size_t neuron_index)
{
    std::vector<std::reference_wrapper<synapse_traits::synapse_parameters<Synapse>>> result;
    for (auto &projection : projections)
    {
        auto synapses =
            projection.get().find_synapses(neuron_index, core::Projection<Synapse>::Search::by_postsynaptic);
        std::transform(
            synapses.begin(), synapses.end(), std::back_inserter(result),
            [&projection](auto const &index)
            { return std::reference_wrapper(std::get<core::synapse_data>(projection.get()[index])); });
    }
    return result;
}


/**
 * @brief Update spike sequence state for the neuron. It's called after a neuron sends a spike.
 * @tparam Neuron Base neuron type.
 * @param neuron Neuron parameters.
 * @param step Current step.
 * @return New state.
 */
template <class Neuron>
neuron_traits::ISIPeriodType update_isi(
    neuron_traits::neuron_parameters<neuron_traits::SynapticResourceSTDPNeuron<Neuron>> &neuron, uint64_t step)
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
 * @brief Check if point is in interval.
 * @param interval_begin Interval start.
 * @param interval_end Interval end.
 * @param point Point that is supposed to be checked.
 * @return Is point in interval.
 */
inline bool is_point_in_interval(uint64_t interval_begin, uint64_t interval_end, uint64_t point)
{
    if (interval_begin > interval_end) std::swap(interval_begin, interval_end);
    return (interval_begin <= point) && (point <= interval_end);
}


/**
 * @brief Distribute neurons resources amongst all synapses.
 * @note If a neuron resource is greater than `1` or `-1` it should be distributed among all synapses.
 * @tparam Neuron Neuron type.
 * @tparam Synapse Synapse type.
 * @param working_projections List of connected to population projections.
 * @param population Population.
 * @param step Current step.
 */
template <class Neuron, class Synapse>
inline void renormalize_resource(
    std::vector<std::reference_wrapper<knp::core::Projection<Synapse>>> &working_projections,
    knp::core::Population<Neuron> &population, uint64_t step)
{
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

        auto synapse_params = get_all_connected_synapses<Synapse>(working_projections, neuron_index);

        // Divide free resource between all synapses.
        auto add_resource_value =
            neuron.free_synaptic_resource_ / (synapse_params.size() + neuron.resource_drain_coefficient_);

        for (auto &synapse : synapse_params)
        {
            synapse.get().rule_.synaptic_resource_ += add_resource_value;
        }

        neuron.free_synaptic_resource_ = 0.0F;
        recalculate_synapse_weights(synapse_params);
    }
}

}  //namespace knp::backends::cpu::populations::impl::training::stdp
