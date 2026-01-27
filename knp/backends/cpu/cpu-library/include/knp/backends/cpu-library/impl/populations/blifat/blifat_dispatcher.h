/**
 * @file blifat_dispatcher.h
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

#include <vector>

#include "blifat_impl.h"


namespace knp::backends::cpu::populations::impl
{

/**
 * @brief Calculate pre impact state of single neuron.
 * @param neuron Neuron.
 */
inline void calculate_pre_impact_single_neuron_state_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> &neuron)
{
    blifat::calculate_pre_impact_single_neuron_state_impl(neuron);
}


/**
 * @brief Calculate pre impact state of single neuron.
 * @param neuron Neuron.
 */
inline void calculate_pre_impact_single_neuron_state_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &neuron)
{
    blifat::calculate_pre_impact_single_neuron_state_impl(neuron);
}


/**
 * @brief Impact neuron.
 * @param neuron Neuron.
 * @param impact Impact message.
 * @param is_forcing Is impact forced.
 */
inline void impact_neuron_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    blifat::impact_neuron_impl(neuron, impact, is_forcing);
}


/**
 * @brief Impact neuron.
 * @param neuron Neuron.
 * @param impact Impact message.
 * @param is_forcing Is impact forced.
 */
inline void impact_neuron_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &neuron,
    const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    blifat::impact_neuron_impl(neuron, impact, is_forcing);
}


/**
 * @brief Calculate post impact state of single neuron.
 * @param neuron Neuron.
 * @return Should neuron produce spike or should not.
 */
inline bool calculate_post_impact_single_neuron_state_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> &neuron)
{
    return blifat::calculate_post_impact_single_neuron_state_impl(neuron);
}


/**
 * @brief Calculate post impact state of single neuron.
 * @param neuron Neuron.
 * @return Should neuron produce spike or should not.
 */
inline bool calculate_post_impact_single_neuron_state_dispatch(
    knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &neuron)
{
    return blifat::calculate_post_impact_single_neuron_state_impl(neuron);
}


/**
 * @brief Train population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
inline void train_population_dispatch(
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::DeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
}


/**
 * @brief Train population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
inline void train_population_dispatch(
    knp::core::Population<knp::neuron_traits::BLIFATNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
}


/**
 * @brief Train population.
 * @param population Population.
 * @param projections Connected projections.
 * @param message Spiking neurons in population at current step.
 * @param step Step.
 */
inline void train_population_dispatch(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
    std::vector<knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> *> const &projections,
    const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
    blifat::train_population_impl(population, projections, message, step);
}

}  //namespace knp::backends::cpu::populations::impl
