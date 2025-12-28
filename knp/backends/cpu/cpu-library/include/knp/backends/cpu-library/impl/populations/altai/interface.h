/**
 * @file interface.h
 * @kaspersky_support Postnikov D.
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

// This include does not work at the end of the day. But its here so code analyzer will work properly.
#include <knp/backends/cpu-library/impl/populations/interface_fwd.h>

#include <vector>

#include "impl.h"

namespace knp::backends::cpu::populations::impl
{

/**
 * @brief Calculate pre impact state of single neuron.
 * @param neuron Neuron.
 */
template <>
inline void calculate_pre_impact_single_neuron_state_interface<altai::AltaiNeuron>(
    knp::neuron_traits::neuron_parameters<altai::AltaiNeuron> &neuron)
{
    altai::calculate_pre_impact_single_neuron_state_impl(neuron);
}

/**
 * @brief Calculate pre impact state of single neuron.
 * @param neuron Neuron.
 */
template <>
inline void calculate_pre_impact_single_neuron_state_interface<altai::STDPAltaiNeuron>(
    knp::neuron_traits::neuron_parameters<altai::STDPAltaiNeuron> &neuron)
{
    altai::calculate_pre_impact_single_neuron_state_impl(neuron);
}

/**
 * @brief Impact neuron.
 * @param neuron Neuron.
 * @param impact Impact message.
 * @param is_forcing Is impact forced.
 */
template <>
inline void impact_neuron_interface<altai::AltaiNeuron>(
    knp::neuron_traits::neuron_parameters<altai::AltaiNeuron> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    altai::impact_neuron_impl(neuron, impact, is_forcing);
}

/**
 * @brief Impact neuron.
 * @param neuron Neuron.
 * @param impact Impact message.
 * @param is_forcing Is impact forced.
 */
template <>
inline void impact_neuron_interface<altai::STDPAltaiNeuron>(
    knp::neuron_traits::neuron_parameters<altai::STDPAltaiNeuron> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    altai::impact_neuron_impl(neuron, impact, is_forcing);
}

/**
 * @brief Calculate post impact state of single neuron.
 * @param neuron Neuron.
 * @return Should neuron produce spike or should not.
 */
template <>
inline bool calculate_post_impact_single_neuron_state_interface<altai::AltaiNeuron>(
    knp::neuron_traits::neuron_parameters<altai::AltaiNeuron> &neuron)
{
    return altai::calculate_post_impact_single_neuron_state_impl(neuron);
}

/**
 * @brief Calculate post impact state of single neuron.
 * @param neuron Neuron.
 * @return Should neuron produce spike or should not.
 */
template <>
inline bool calculate_post_impact_single_neuron_state_interface<altai::STDPAltaiNeuron>(
    knp::neuron_traits::neuron_parameters<altai::STDPAltaiNeuron> &neuron)
{
    return altai::calculate_post_impact_single_neuron_state_impl(neuron);
}

/**
 * @brief Teach population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
template <>
inline void teach_population_interface<altai::AltaiNeuron, knp::synapse_traits::DeltaSynapse>(
    knp::core::Population<altai::AltaiNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::DeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
}

/**
 * @brief Teach population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
template <>
inline void teach_population_interface<altai::AltaiNeuron, knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>(
    knp::core::Population<altai::AltaiNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
}

/**
 * @brief Teach population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
template <>
inline void teach_population_interface<altai::STDPAltaiNeuron, knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>(
    knp::core::Population<altai::STDPAltaiNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
    altai::teach_population_impl(population, projections, message, step);
}


}  //namespace knp::backends::cpu::populations::impl
