/**
 * @file populations.h
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
 * @brief Namespace for CPU backend's populations.
 */
namespace knp::backends::cpu::populations
{

/**
 * @brief Partially calculate population before it receives synaptic impact messages.
 * @param population Population to update.
 * @param start Index of the first neuron to calculate.
 * @param end Index of the last neuron to calculate.
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
 * @brief Impact population.
 * @param population Population.
 * @param messages Impact messages.
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
 * @param population population to update.
 * @param message output spike message to update.
 * @param start index of the first neuron to update.
 * @param end index of the last neuron to update.
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
 * @brief Train population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
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
