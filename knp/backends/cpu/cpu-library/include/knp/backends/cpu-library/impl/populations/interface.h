/**
 * @file interface.h
 * @kaspersky_support Postnikov D.
 * @date 02.12.2025
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

#include <vector>

#include "altai/interface.h"
#include "blifat/interface.h"

namespace knp::backends::cpu::populations::impl
{

/**
 * @brief Calculate pre impact state of single neuron.
 * @param neuron Neuron.
 */
template <class Neuron>
void calculate_pre_impact_single_neuron_state_interface(knp::neuron_traits::neuron_parameters<Neuron> &neuron)
{
    throw std::runtime_error("Unsupported neuron type");
}

/**
 * @brief Impact neuron.
 * @param neuron Neuron.
 * @param impact Impact message.
 * @param is_forcing Is impact forced.
 */
template <class Neuron>
void impact_neuron_interface(
    knp::neuron_traits::neuron_parameters<Neuron> &neuron, const knp::core::messaging::SynapticImpact &impact,
    bool is_forcing)
{
    throw std::runtime_error("Unsupported neuron type");
}

/**
 * @brief Calculate post impact state of single neuron.
 * @param neuron Neuron.
 * @return Should neuron produce spike or should not.
 */
template <class Neuron>
bool calculate_post_impact_single_neuron_state_interface(knp::neuron_traits::neuron_parameters<Neuron> &neuron)
{
    throw std::runtime_error("Unsupported neuron type");
}

/**
 * @brief Teach population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
template <class Neuron, class Synapse>
void teach_population_interface(
    knp::core::Population<Neuron> &population, std::vector<knp::core::Projection<Synapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
    throw std::runtime_error("Unsupported neuron-synapse pair type combination");
}

}  // namespace knp::backends::cpu::populations::impl
