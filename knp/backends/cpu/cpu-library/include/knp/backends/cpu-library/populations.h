/**
 * @file populations.h
 * @brief Interface for working with populations.
 * @kaspersky_support Postnikov D.
 * @date 26.11.2025
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

#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <spdlog/spdlog.h>

#include <vector>

#include "impl/populations/population_dispatcher.h"


/**
 * @brief Namespace for CPU backend population functions.
 */
namespace knp::backends::cpu::populations
{

/**
 * @brief Partially calculate population before it receives synaptic impact messages.
 * 
 * @note The 'end' parameter is exclusive, that is a neuron with the specified 'end' index is not calculated. 
 * 
 * @tparam Neuron type of neurons stored in the population.
 * 
 * @param population population to calculate.
 * @param start index of the first neuron to calculate.
 * @param end index of the last neuron to calculate.
 * 
 */
template <class Neuron>
void calculate_pre_impact_population_state(knp::core::Population<Neuron> &population, size_t start, size_t end)
{
    SPDLOG_TRACE("Calculate pre impact state of [{},{}] neurons.", start, end);
    for (size_t i = start; i < end; ++i)
    {
        impl::calculate_pre_impact_single_neuron_state_dispatch(population[i]);
    }
}


/**
 * @brief Dispatch synaptic impact messages to population.
 * 
 * @tparam Neuron type of neurons stored in the population.
 * 
 * @param population population to impact.
 * @param messages synaptic impact messages to dispatch.
 * 
 */
template <class Neuron>
void impact_population(
    knp::core::Population<Neuron> &population, const std::vector<core::messaging::SynapticImpactMessage> &messages)
{
    SPDLOG_TRACE("Impact population.");
    for (const auto &message : messages)
    {
        for (const auto &impact : message.impacts_)
        {
            impl::impact_neuron_dispatch(population[impact.postsynaptic_neuron_index_], impact, message.is_forcing_);
        }
    }
}


/**
 * @brief Partially calculate population after it receives synaptic impact messages.
 * 
 * @note The 'end' parameter is exclusive, that is a neuron with the specified 'end' index is not calculated. 
 * 
 * @details The function iterates over the neurons in the range `[start, end)`. If a neuron produces a spike, then 
 * its index is appended to the message defined in the @p message parameter.
 * 
 * @tparam Neuron type of neurons stored in the population.
 * 
 * @param population population to calculate.
 * @param message output spike message to update.
 * @param start index of the first neuron to calculate.
 * @param end index of the last neuron to calculate.
 * 
 */
template <class Neuron>
void calculate_post_impact_population_state(
    knp::core::Population<Neuron> &population, knp::core::messaging::SpikeMessage &message, size_t start, size_t end)
{
    SPDLOG_TRACE("Calculate post impact state of [{},{}] neurons.", start, end);
    for (size_t i = start; i < end; ++i)
    {
        if (impl::calculate_post_impact_single_neuron_state_dispatch(population[i]))
        {
            message.neuron_indexes_.push_back(i);
        }
    }
}


/**
 * @brief Train the population for the given simulation step.
 * 
 * @tparam Neuron type of neurons stored in the population.
 * @tparam Synapse type of synapses used in the connected projections.
 * 
 * @param population population to train.
 * @param projections connected projections that send synaptic impacts.
 * @param message spiking neurons in the population at the current step.
 * @param step simulation step for which the training is performed.
 * 
 */
template <class Neuron, class Synapse>
void train_population(
    knp::core::Population<Neuron> &population,
    std::vector<std::reference_wrapper<knp::core::Projection<Synapse>>> &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
    SPDLOG_TRACE("Training population.");
    impl::train_population_dispatch(population, projections, message, step);
}

}  // namespace knp::backends::cpu::populations
