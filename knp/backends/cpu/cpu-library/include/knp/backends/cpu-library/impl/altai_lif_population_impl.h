/**
 * @file altai_lif_population_impl.h
 * @kaspersky_support Vartenkov A.
 * @date 01.04.2025
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
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>

#include <algorithm>
#include <vector>

#include "synaptic_resource_stdp_impl.h"

/**
 * @brief CPU backend namespace.
 */
namespace knp::backends::cpu
{


/**
 * @brief LIF neuron reaction to input types.
 * @tparam BasicLifNeuron LIF neuron type.
 * @param neuron the neuron receiving the impact.
 * @param synapse_type synapse type.
 * @param impact_value synaptic impact value.
 */
template <class BasicLifNeuron>
void impact_neuron(
    knp::neuron_traits::neuron_parameters<BasicLifNeuron> &neuron, const knp::synapse_traits::OutputType &synapse_type,
    float impact_value)
{
    switch (synapse_type)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            neuron.potential_ += impact_value;
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            neuron.potential_ -= impact_value;
            break;
        case knp::synapse_traits::OutputType::DOPAMINE:
            neuron.dopamine_value_ += impact_value;
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            neuron.total_blocking_period_ = static_cast<unsigned int>(impact_value);
            break;
        default:
            break;
    }
}

template <class Neuron>
constexpr bool has_dopamine_plasticity_altai()
{
    return false;
}

template <>
constexpr bool has_dopamine_plasticity_altai<neuron_traits::SynapticResourceSTDPAltAILIFNeuron>()
{
    return true;
}
/**
 * @brief Calculate neuron state before it starts accepting inputs.
 * @tparam BasicLifNeuron LIF neuron type.
 * @param population the population that is being processed.
 */
template <class BasicLifNeuron>
void calculate_pre_input_state_lif(knp::core::Population<BasicLifNeuron> &population)
{
    for (auto &neuron : population)
    {
        neuron.potential_ = std::round(neuron.potential_);

        neuron.potential_ = neuron.do_not_save_ ? static_cast<float>(neuron.potential_reset_value_) : neuron.potential_;

        if constexpr (has_dopamine_plasticity_altai<BasicLifNeuron>())
        {
            neuron.dopamine_value_ = 0.0;
            //not sure
            //neuron.free_synaptic_resource_ = 0.0;
            neuron.is_being_forced_ = false;
        }
        neuron.pre_impact_potential_ = neuron.potential_;
    }
}


template <class BasicLifNeuron>
void leak_potential(knp::core::Population<BasicLifNeuron> &population)
{
    for (auto &neuron : population)
    {
        // -1 if leak_rev is true and potential < 0, 1 otherwise.
        const int sign = (neuron.leak_rev_ && neuron.potential_ < 0) ? -1 : 1;
        neuron.potential_ += neuron.potential_leak_ * sign;
    }
}

/**
 * @brief Single-threaded neuron impact processing
 * @tparam BasicLifNeuron neuron type.
 * @param population population to be impacted.
 * @param messages vector of synapse impacts.
 */
template <class BasicLifNeuron>
void process_inputs_lif(
    knp::core::Population<BasicLifNeuron> &population,
    const std::vector<knp::core::messaging::SynapticImpactMessage> &messages)
{
    for (const auto &msg : messages)
    {
        for (const auto &impact : msg.impacts_)
        {
            auto &neuron = population[impact.postsynaptic_neuron_index_];
            impact_neuron(neuron, impact.synapse_type_, impact.impact_value_);
            //population[impact.postsynaptic_neuron_index_].potential_ += impact.impact_value_;

            //not sure if we need this, just pasted this from blifat.
            if constexpr (has_dopamine_plasticity_altai<BasicLifNeuron>())
            {
                if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
                {
                    neuron.is_being_forced_ |= msg.is_forcing_;
                }
            }
        }
    }
    leak_potential(population);
}

/**
 * @brief Calculate which neurons should emit spikes.
 * @tparam BasicLifNeuron LIF neuron type.
 * @param population the population that is being processed.
 * @return spiked neuron indexes.
 */
template <class BasicLifNeuron>
knp::core::messaging::SpikeData calculate_spikes_lif(knp::core::Population<BasicLifNeuron> &population)
{
    knp::core::messaging::SpikeData spikes;
    for (knp::core::messaging::SpikeIndex i = 0; i < population.size(); ++i)
    {
        auto &neuron = population[i];

        //PASTED FROM BLIFAT
        if (neuron.total_blocking_period_ <= 0)
        {
            // Restore potential that the neuron had before impacts.
            neuron.potential_ = neuron.pre_impact_potential_;
            bool was_negative = neuron.total_blocking_period_ < 0;
            // If it is negative, increase by 1.
            neuron.total_blocking_period_ += was_negative;
            // If it is now zero, but was negative before, increase it to max, else leave it as is.
            neuron.total_blocking_period_ +=
                std::numeric_limits<int64_t>::max() * ((neuron.total_blocking_period_ == 0) && was_negative);
        }
        else
        {
            neuron.total_blocking_period_ -= 1;
        }

        bool was_reset = false;

        if (neuron.potential_ >= neuron.activation_threshold_ + neuron.additional_threshold_)
        {
            spikes.push_back(i);
            if (spikes.size() > 20)
            {
                std::cout << "aaa" << std::endl;
            }
            if (neuron.is_diff_) neuron.potential_ -= neuron.activation_threshold_ + neuron.additional_threshold_;
            if (neuron.is_reset_)
            {
                neuron.potential_ = neuron.potential_reset_value_;
                was_reset = true;
            }
        }
        if (neuron.potential_ <= -static_cast<float>(neuron.negative_activation_threshold_) && !was_reset)
        {
            // Might probably want a negative spike, but we don't have any of the sort in KNP. Not a large problem,
            // just requires some conversion.
            if (neuron.saturate_)
            {
                neuron.potential_ = -static_cast<float>(neuron.negative_activation_threshold_);
                continue;
            }
            if (neuron.is_reset_)
                neuron.potential_ = -neuron.potential_reset_value_;
            else if (neuron.is_diff_)
                neuron.potential_ += neuron.negative_activation_threshold_;
        }
    }

    return spikes;
}

/**
 * @brief Common calculation algorithm for all LIF-like neurons.
 * @tparam BasicLifNeuron LIF neuron type.
 * @param population population of LIF neurons.
 * @param messages input messages.
 * @return spiked neuron indexes.
 */
template <class BasicLifNeuron>
knp::core::messaging::SpikeData calculate_lif_population_data(
    knp::core::Population<BasicLifNeuron> &population,
    const std::vector<knp::core::messaging::SynapticImpactMessage> &messages)
{
    calculate_pre_input_state_lif(population);
    process_inputs_lif(population, messages);
    knp::core::messaging::SpikeData spikes = calculate_spikes_lif(population);
    return spikes;
}


/**
 * @brief Full LIF population step calculation.
 * @tparam BasicLifNeuron LIF neuron type.
 * @param population population to be processed.
 * @param endpoint endpoint for message exchange.
 * @param step_n current step.
 * @return spike message if it was emitted.
 * @note This function emits the message it returns, you don't need to do it again.
 */
template <class BasicLifNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_lif_population_impl(
    knp::core::Population<BasicLifNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    std::vector<knp::core::messaging::SynapticImpactMessage> messages =
        endpoint.unload_messages<knp::core::messaging::SynapticImpactMessage>(population.get_uid());
    knp::core::messaging::SpikeMessage message_out{{population.get_uid(), step_n}, {}};
    message_out.neuron_indexes_ = calculate_lif_population_data(population, messages);
    if (message_out.neuron_indexes_.empty())
    {
        return {};
    }
    endpoint.send_message(message_out);
    return message_out;
}


template <>
inline void process_spiking_neurons<knp::neuron_traits::AltAILIF>(
    const core::messaging::SpikeMessage &msg,
    std::vector<StdpProjection<synapse_traits::DeltaSynapse> *> &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron> &population, uint64_t step)
{
    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    // It's very important that during this function no projection invalidates iterators.
    // Loop over neurons.
    for (const auto &spiked_neuron_index : msg.neuron_indexes_)
    {
        auto synapse_params = get_all_connected_synapses<SynapseType>(working_projections, spiked_neuron_index);
        auto &neuron = population[spiked_neuron_index];
        neuron.last_spike_step_ = step;
        // Calculate neuron ISI status.
        update_isi<knp::neuron_traits::AltAILIF>(neuron, step);
        if (neuron_traits::ISIPeriodType::period_started == neuron.isi_status_)
        {
            neuron.stability_ -= neuron.stability_change_at_isi_;
        }
        neuron.additional_threshold_ = 0.0;
        // Mark contributed synapses
        for (auto *synapse : synapse_params)
        {
            neuron.additional_threshold_ += synapse->weight_ * (synapse->weight_ > 0);
            const bool had_spike = is_point_in_interval(
                step - synapse->rule_.dopamine_plasticity_period_, step,
                synapse->rule_.last_spike_step_ + synapse->delay_ - 1);
            // While period continues we don't change has_contributed from true to false.
            if (neuron_traits::ISIPeriodType::period_continued != neuron.isi_status_ || had_spike)
                synapse->rule_.has_contributed_ = had_spike;
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
        recalculate_synapse_weights<knp::synapse_traits::DeltaSynapse>(synapse_params);
    }
}

template <>
inline void do_dopamine_plasticity<knp::neuron_traits::AltAILIF>(
    std::vector<StdpProjection<synapse_traits::DeltaSynapse> *> &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron> &population, uint64_t step)
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
                get_all_connected_synapses<SynapseType>(working_projections, neuron_index);
            // Change synapse values for both `D > 0` and `D < 0`.
            for (auto *synapse : synapse_params)
            {
                // if ((step - synapse->rule_.last_spike_step_ < synapse->rule_.dopamine_plasticity_period_)
                if (step - neuron.last_spike_step_ <= neuron.dopamine_plasticity_time_ &&
                    synapse->rule_.has_contributed_)
                {
                    // Change synapse resource.
                    float d_r =
                        neuron.dopamine_value_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);
                    synapse->rule_.synaptic_resource_ += d_r;
                    neuron.free_synaptic_resource_ -= d_r;
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
                const double difference = step - neuron.first_isi_spike_ - neuron.isi_max_;
                neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
                                     std::max(dopamine_constant - std::fabs(difference) / neuron.isi_max_, -1.0);
            }
            recalculate_synapse_weights(synapse_params);
        }
    }
}

}  // namespace knp::backends::cpu
