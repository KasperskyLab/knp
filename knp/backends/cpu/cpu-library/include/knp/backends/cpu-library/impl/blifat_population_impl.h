/**
 * @file blifat_population_impl.h
 * @brief Definition of BLIFAT neuron calculation routines.
 * @kaspersky_support Artiom N.
 * @date 21.02.2023
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

#include <knp/core/message_bus.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "synaptic_resource_stdp_impl.h"

/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
template <class Neuron>
constexpr bool has_dopamine_plasticity()
{
    return false;
}

template <>
constexpr bool has_dopamine_plasticity<neuron_traits::SynapticResourceSTDPBLIFATNeuron>()
{
    return true;
}


/**
 * @brief Apply STDP to all presynaptic connections of a single population.
 * @tparam NeuronType type of neuron that is compatible with STDP.
 * @param msg spikes emited by population.
 * @param working_projections all projections. The function skips projections that are not connected, locked or are of a
 * wrong type.
 * @param population population.
 * @param step current network step.
 * @note All projections are supposed to be of the same type.
 */
template <class NeuronType>
void process_spiking_neurons(
    const core::messaging::SpikeMessage &msg,
    std::vector<knp::core::Projection<
        knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, synapse_traits::DeltaSynapse>> *>
        &working_projections,
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPNeuron<NeuronType>> &population, uint64_t step);


/**
 * @brief Calculate the result of a synaptic impact on a neuron.
 * @tparam BlifatLikeNeuron type of neuron which inference can be calculated as for a BLIFAT neuron.
 * @param neuron BLIFAT-like neuron parameters.
 * @param synapse_type type of input signal.
 * @param impact_value value of input signal.
 * @note We might want to impact a neuron with a whole message if it continues to have shared values.
 */
template <class BlifatLikeNeuron>
void impact_neuron(
    typename knp::core::Population<BlifatLikeNeuron>::NeuronParameters &neuron,
    const knp::synapse_traits::OutputType &synapse_type, float impact_value)
{
    switch (synapse_type)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            neuron.potential_ += impact_value;
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            neuron.potential_ -= impact_value;
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
            neuron.inhibitory_conductance_ += impact_value;
            break;
        case knp::synapse_traits::OutputType::DOPAMINE:
            neuron.dopamine_value_ += impact_value;
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            neuron.total_blocking_period_ = static_cast<unsigned int>(impact_value);
            break;
    }
}


/**
 * @brief Process messages sent to the current population.
 * @param population population to update.
 * @param messages synaptic impact messages sent to the population.
 * @note The method is used for parallelization. See later if this method serves as a bottleneck.
 */
template <class BlifatLikeNeuron>
void process_inputs(
    knp::core::Population<BlifatLikeNeuron> &population,
    const std::vector<core::messaging::SynapticImpactMessage> &messages)
{
    SPDLOG_TRACE("Process inputs.");
    for (const auto &message : messages)
    {
        for (const auto &impact : message.impacts_)
        {
            auto &neuron = population[impact.postsynaptic_neuron_index_];
            impact_neuron<BlifatLikeNeuron>(neuron, impact.synapse_type_, impact.impact_value_);
            if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
            {
                if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
                {
                    neuron.is_being_forced_ |= message.is_forcing_;
                }
            }
        }
    }
}


/**
 * @brief Calculate a single neuron state before impacts.
 * @tparam BlifatLikeNeuron type of neuron which inference can be calculated as for a BLIFAT neuron.
 * @param neuron neuron parameters.
 */
template <class BlifatLikeNeuron>
void calculate_single_neuron_state(typename knp::core::Population<BlifatLikeNeuron>::NeuronParameters &neuron)
{
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;
    if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
    {
        neuron.dopamine_value_ = 0.0;
        neuron.is_being_forced_ = false;
    }

    if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
    {
        neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
    }
    else
    {
        neuron.potential_ *= neuron.potential_decay_;
    }
    neuron.pre_impact_potential_ = neuron.potential_;
}


/**
 * @brief Partially calculate population before it receives synaptic impact messages.
 * @param population population to update.
 * @param part_start index of the first neuron to calculate.
 * @param part_size number of neurons to calculate in a single call.
 * @note The method if used for parallelization.
 */
template <class BlifatLikeNeuron>
void calculate_neurons_state_part(
    knp::core::Population<BlifatLikeNeuron> &population, size_t part_start, size_t part_size)
{
    size_t part_end = std::min(part_start + part_size, population.size());
    SPDLOG_TRACE("Calculate neuron state part.");
    for (size_t i = part_start; i < part_end; ++i)
    {
        auto &neuron = population[i];
        ++neuron.n_time_steps_since_last_firing_;
        calculate_single_neuron_state<BlifatLikeNeuron>(neuron);
    }
}


/**
 *
 * @tparam BlifatLikeNeuron type of neuron which inference can be calculated as for a BLIFAT neuron.
 * @param population neurons population.
 * @param messages messages from the projection to the populations.
 */
template <class BlifatLikeNeuron>
void calculate_neurons_state(
    knp::core::Population<BlifatLikeNeuron> &population,
    const std::vector<core::messaging::SynapticImpactMessage> &messages)
{
    calculate_neurons_state_part(population, 0, population.size());
    process_inputs(population, messages);
}


template <class BlifatLikeNeuron>
bool calculate_neuron_post_input_state(typename knp::core::Population<BlifatLikeNeuron>::NeuronParameters &neuron)
{
    bool spike = false;
    if (neuron.total_blocking_period_ <= 0)
    {
        // TODO: Make it more readable, don't be afraid to use if operators.
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

    if (neuron.inhibitory_conductance_ < 1.0)
    {
        neuron.potential_ -=
            (neuron.potential_ - neuron.reversal_inhibitory_potential_) * neuron.inhibitory_conductance_;
    }
    else
    {
        neuron.potential_ = neuron.reversal_inhibitory_potential_;
    }

    if ((neuron.n_time_steps_since_last_firing_ > neuron.absolute_refractory_period_) &&
        (neuron.potential_ >= neuron.activation_threshold_ + neuron.dynamic_threshold_ + neuron.additional_threshold_))
    {
        SPDLOG_TRACE("Neuron spiked.");
        // Spike.
        neuron.dynamic_threshold_ += neuron.threshold_increment_;
        neuron.postsynaptic_trace_ += neuron.postsynaptic_trace_increment_;

        neuron.potential_ = neuron.potential_reset_value_;
        neuron.bursting_phase_ = neuron.bursting_period_;
        neuron.n_time_steps_since_last_firing_ = 0;
        spike = true;
    }

    if (neuron.potential_ < neuron.min_potential_)
    {
        neuron.potential_ = neuron.min_potential_;
    }

    return spike;
}


/**
 * @brief Finish calculation after the neurons get synaptic impacts.
 * @tparam BlifatLikeNeuron type of neuron which inference can be calculated as for a BLIFAT neuron.
 * @param population population of BLIFAT-like neurons.
 * @param neuron_indexes output parameter, indexes of spiked neurons.
 */
template <class BlifatLikeNeuron>
void calculate_neurons_post_input_state(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::messaging::SpikeData &neuron_indexes)
{
    SPDLOG_TRACE("Calculate neuron post-input state part.");
    for (size_t index = 0; index < population.size(); ++index)
    {
        if (calculate_neuron_post_input_state<BlifatLikeNeuron>(population[index]))
        {
            neuron_indexes.push_back(index);
        }
    }
}


/**
 * @brief Partially calculate population after it receives synaptic impact messages.
 * @param population population to update.
 * @param message output spike message to update.
 * @param part_start index of the first neuron to update.
 * @param part_size number of neurons to calculate in a single call.
 * @param mutex mutex that is locked to update a message.
 * @note This method is used for parallelization.
 */
template <class BlifatLikeNeuron>
void calculate_neurons_post_input_state_part(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::messaging::SpikeMessage &message, size_t part_start,
    size_t part_size, std::mutex &mutex)
{
    SPDLOG_TRACE("Calculate neuron post-input state part.");
    size_t part_end = std::min(part_start + part_size, population.size());
    std::vector<size_t> output;
    for (size_t i = part_start; i < part_end; ++i)
    {
        if (calculate_neuron_post_input_state<BlifatLikeNeuron>(population[i]))
        {
            output.push_back(i);
        }
    }

    // Updating common neuron indexes.
    const std::lock_guard<std::mutex> lock(mutex);
    message.neuron_indexes_.reserve(message.neuron_indexes_.size() + output.size());
    message.neuron_indexes_.insert(message.neuron_indexes_.end(), output.begin(), output.end());
}


/**
 * @brief Process BLIFAT neuron population and return spiked neuron indexes.
 * @tparam BlifatLikeNeuron type of neuron which inference can be calculated the same as BLIFAT.
 * @param population population of BLIFAT-like neurons.
 * @param endpoint message endpoint.
 * @return indexes of spiked neurons.
 */
template <class BlifatLikeNeuron>
knp::core::messaging::SpikeData calculate_blifat_population_data(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::MessageEndpoint &endpoint)
{
    SPDLOG_DEBUG("Calculating BLIFAT population {}...", std::string{population.get_uid()});
    // This whole function might be optimizable if we find a way to not loop over the whole population.
    std::vector<core::messaging::SynapticImpactMessage> messages =
        endpoint.unload_messages<core::messaging::SynapticImpactMessage>(population.get_uid());

    calculate_neurons_state(population, messages);
    knp::core::messaging::SpikeData neuron_indexes;
    calculate_neurons_post_input_state(population, neuron_indexes);

    return neuron_indexes;
}


template <class BlifatLikeNeuron>
std::optional<core::messaging::SpikeMessage> calculate_blifat_population_impl(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    auto neuron_indexes{calculate_blifat_population_data(population, endpoint)};
    std::optional<knp::core::messaging::SpikeMessage> message_opt = {};
    if (!neuron_indexes.empty())
    {
        knp::core::messaging::SpikeMessage res_message{{population.get_uid(), step_n}, neuron_indexes};
        endpoint.send_message(res_message);
        SPDLOG_DEBUG("Sent {} spike(s).", res_message.neuron_indexes_.size());
        message_opt = std::move(res_message);
    }
    return message_opt;
}


/**
 * @brief Make one execution step for a population of BLIFAT neurons.
 * @param population population to update.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @param mutex mutex.
 * @return indexes of spiked neurons.
 */
template <class BlifatLikeNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_blifat_population_impl(
    knp::core::Population<BlifatLikeNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n,
    std::mutex &mutex)
{
    auto neuron_indexes{calculate_blifat_population_data(population, endpoint)};
    if (!neuron_indexes.empty())
    {
        knp::core::messaging::SpikeMessage res_message{{population.get_uid(), step_n}, neuron_indexes};
        const std::lock_guard<std::mutex> guard(mutex);
        endpoint.send_message(res_message);
        SPDLOG_DEBUG("Sent {} spike(s).", res_message.neuron_indexes_.size());
        return res_message;
    }
    return {};
}

}  // namespace knp::backends::cpu
