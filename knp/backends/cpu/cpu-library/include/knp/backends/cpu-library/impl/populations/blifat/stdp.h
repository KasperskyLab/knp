/**
 * @file stdp.h
 * @kaspersky_support Postnikov D.
 * @date 12.12.2025
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

#include <knp/backends/cpu-library/impl/populations/training/stdp.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>

#include <algorithm>
#include <vector>


namespace knp::backends::cpu::populations::impl::blifat
{

template <typename Synapse>
inline void process_spiking_neurons_impl(
    const core::messaging::SpikeMessage &msg, std::vector<knp::core::Projection<Synapse> *> const &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population, uint64_t step)
{
    // It's very important that during this function no projection invalidates iterators.
    // Loop over neurons.
    for (const auto &spiked_neuron_index : msg.neuron_indexes_)
    {
        auto synapse_params =
            training::stdp::get_all_connected_synapses<Synapse>(working_projections, spiked_neuron_index);
        auto &neuron = population[spiked_neuron_index];
        neuron.last_spike_step_ = step;
        // Calculate neuron ISI status.
        training::stdp::update_isi<knp::neuron_traits::BLIFATNeuron>(neuron, step);
        if (neuron_traits::ISIPeriodType::period_started == neuron.isi_status_)
            neuron.stability_ -= neuron.stability_change_at_isi_;
        neuron.additional_threshold_ = 0.0;
        // Mark contributed synapses
        for (auto *synapse : synapse_params)
        {
            neuron.additional_threshold_ += synapse->weight_ * (synapse->weight_ > 0);
            const bool had_spike = training::stdp::is_point_in_interval(
                step - synapse->rule_.dopamine_plasticity_period_, step,
                synapse->rule_.last_spike_step_ + synapse->delay_ - 1);
            // While period continues we don't change has_contributed from true to false.
            if (neuron_traits::ISIPeriodType::period_continued != neuron.isi_status_ || had_spike)
            {
                synapse->rule_.has_contributed_ = had_spike;
            }
        }
        neuron.additional_threshold_ *= neuron.synapse_sum_threshold_coefficient_;

        // This is a new spiking sequence, we can update synapses now.
        if (neuron.isi_status_ != neuron_traits::ISIPeriodType::period_continued)
        {
            for (auto *synapse : synapse_params)
            {
                synapse->rule_.had_hebbian_update_ = false;
            }
        }

        // Update synapse-only data.
        if (neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced)
        {
            for (auto *synapse : synapse_params)
            {
                // Unconditional decreasing synaptic resource.
                // TODO: NOT HERE. This shouldn't matter now as d_u_ is zero for our task, but the logic is wrong.
                synapse->rule_.synaptic_resource_ -= synapse->rule_.d_u_;
                neuron.free_synaptic_resource_ += synapse->rule_.d_u_;
                // Hebbian plasticity.
                // 1. Check if synapse ever got a spike in the current ISI period.
                if (synapse->rule_.has_contributed_ && !synapse->rule_.had_hebbian_update_)
                {
                    // 2. If it did, then update synaptic resource value.
                    const float d_h = neuron.d_h_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);

                    synapse->rule_.synaptic_resource_ += d_h;
                    neuron.free_synaptic_resource_ -= d_h;
                    synapse->rule_.had_hebbian_update_ = true;
                }
            }
        }
        // Recalculating synapse weights. Sometimes it probably doesn't need to happen, check it later.
        training::stdp::recalculate_synapse_weights<knp::synapse_traits::DeltaSynapse>(synapse_params);
    }
}


template <typename Synapse>
inline void do_dopamine_plasticity_impl(
    std::vector<knp::core::Projection<Synapse> *> const &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population, uint64_t step)
{
    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    using SynapseParamType = knp::synapse_traits::synapse_parameters<SynapseType>;
    for (size_t neuron_index = 0; neuron_index < population.size(); ++neuron_index)
    {
        auto &neuron = population[neuron_index];
        // Dopamine processing. Dopamine punishment if forced does nothing.
        if (neuron.dopamine_value_ > 0.0 ||
            (neuron.dopamine_value_ < 0.0 && neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced))
        {
            std::vector<SynapseParamType *> synapse_params =
                training::stdp::get_all_connected_synapses<SynapseType>(working_projections, neuron_index);
            // Change synapse values for both `D > 0` and `D < 0`.
            for (auto *synapse : synapse_params)
            {
                // if ((step - synapse->rule_.last_spike_step_ < synapse->rule_.dopamine_plasticity_period_)
                if (step - neuron.last_spike_step_ <= neuron.dopamine_plasticity_time_ &&
                    synapse->rule_.has_contributed_)
                {
                    // Change synapse resource.
                    float resource_change =
                        neuron.dopamine_value_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);

                    synapse->rule_.synaptic_resource_ += resource_change;
                    neuron.free_synaptic_resource_ -= resource_change;
                }
            }
            // Stability changes.
            if (neuron.is_being_forced_ || neuron.dopamine_value_ < 0)
            {
                // A dopamine reward when forced or a dopamine punishment reduce stability by `r * D`.
                neuron.stability_ -= neuron.dopamine_value_ * neuron.stability_change_parameter_;
                neuron.stability_ = std::max(neuron.stability_, 0.0F);
            }
            else
            {
                // A dopamine reward when non-forced changes stability by `D max(2 - |t(TSS) - ISImax| / ISImax, -1)`.
                const double dopamine_constant = 2.0;
                const double difference = std::fabs(step - neuron.first_isi_spike_ - neuron.isi_max_);
                neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
                                     std::max(dopamine_constant - difference / neuron.isi_max_, -1.0);
            }
            training::stdp::recalculate_synapse_weights(synapse_params);
        }
    }
}


inline void train_population_impl(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
    if (message.neuron_indexes_.size())
    {
        process_spiking_neurons_impl(message, projections, population, step);
    }

    do_dopamine_plasticity_impl(projections, population, step);

    training::stdp::renormalize_resource(projections, population, step);
}

}  //namespace knp::backends::cpu::populations::impl::blifat
