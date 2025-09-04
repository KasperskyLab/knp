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

#include <mutex>
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
        default:
            break;
    }
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
            population[impact.postsynaptic_neuron_index_].potential_ += impact.impact_value_;
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
        bool was_reset = false;
        auto &neuron = population[i];
        if (neuron.potential_ >= neuron.activation_threshold_)
        {
            spikes.push_back(i);
            if (neuron.is_diff_) neuron.potential_ -= neuron.activation_threshold_;
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
    //TODO
}


template <>
inline void do_dopamine_plasticity<knp::neuron_traits::AltAILIF>(
    std::vector<StdpProjection<synapse_traits::DeltaSynapse> *> &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPAltAILIFNeuron> &population, uint64_t step)
{
    //TODO
}


}  // namespace knp::backends::cpu
