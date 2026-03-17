/**
 * @file connectors.h
 * @brief Population connector.
 * @kaspersky_support Artiom N.
 * @date 21.10.2024
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

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <optional>

#include "creators.h"
#include "synapse_parameters_generators.h"

/**
 * @brief Projection namespace.
 */
namespace knp::framework::projection
{

/**
 * @brief Connect presynaptic and postsynaptic populations.
 * @details The function makes connections between each presynaptic population (source) neuron to each postsynaptic
 * population (destination) neuron.
 * @tparam SourceNeuronType type of the presynaptic population neuron.
 * @tparam DestinationNeuronType type of the postsynaptic population neuron.
 * @tparam SynapseType new projection synapse type.
 * @param src presynaptic population.
 * @param dst postsynaptic population.
 * @param syn_gen synapse parameters generator.
 * @return new projection between two populations.
 */
template <typename SynapseType, typename SourceNeuronType, typename DestinationNeuronType>
[[nodiscard]] typename core::Projection<SynapseType> connect_populations(
    const core::Population<SourceNeuronType> &src, const core::Population<DestinationNeuronType> &dst,
    typename parameters_generators::SynGen2ParamsType<SynapseType> syn_gen =
        parameters_generators::default_synapse_gen_2param<SynapseType>)
{
    return creators::all_to_all<SynapseType>(src.get_uid(), dst.get_uid(), src.size(), dst.size(), syn_gen);
}


/**
 * @brief Connect presynaptic and postsynaptic populations.
 * @details The function makes a projection with arbitrary connections between population neurons.
 * @tparam SourceNeuronType type of the presynaptic population neuron.
 * @tparam DestinationNeuronType type of the postsynaptic population neuron.
 * @tparam SynapseType new projection synapse type.
 * @param src presynaptic population.
 * @param dst postsynaptic population.
 * @param syn_gen synapse parameters generator.
 * @param num_iterations number of iterations.
 * @return new projection between two populations.
 */
template <typename SynapseType, typename SourceNeuronType, typename DestinationNeuronType>
[[nodiscard]] typename core::Projection<SynapseType> connect_populations(
    const core::Population<SourceNeuronType> &src, const core::Population<DestinationNeuronType> &dst,
    typename knp::core::Projection<SynapseType>::SynapseGenerator syn_gen, size_t num_iterations)
{
    return core::Projection<SynapseType>(src.get_uid(), dst.get_uid(), syn_gen, num_iterations);
}

}  // namespace knp::framework::projection
