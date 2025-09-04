/**
 * @file altai_lif_population.h
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
#include <knp/backends/cpu-library/impl/altai_lif_population_impl.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>

#include <optional>


/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{

/**
 * @brief Calculate LIF population step.
 * @tparam LifNeuron LIF neuron type.
 * @param population population to be calculated.
 * @param endpoint endpoint for message exchange.
 * @param step_n current step.
 * @return message if a message was sent by function.
 */
template <class LifNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_lif_population(
    knp::core::Population<LifNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_lif_population_impl(population, endpoint, step_n);
}


/**
 * @brief Make one execution step for a population of `SynapticResourceSTDPNeuron` neurons.
 * @tparam AltAILIFLikeNeuron type of a neuron with AltAILIF-like parameters.
 * @tparam BaseSynapseType base synapse type.
 * @tparam ProjectionContainer type of a projection container.
 * @param population population to update.
 * @param container projection container from backend.
 * @param endpoint message endpoint used for message exchange.
 * @param step_n execution step.
 * @return message containing indexes of spiked neurons.
 */
template <class AltAILIFLikeNeuron, class BaseSynapseType, class ProjectionContainer>
std::optional<core::messaging::SpikeMessage> calculate_resource_stdp_lif_population(
    knp::core::Population<neuron_traits::SynapticResourceSTDPNeuron<AltAILIFLikeNeuron>> &population,
    ProjectionContainer &container, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    using StdpSynapseType = synapse_traits::STDP<synapse_traits::STDPSynapticResourceRule, BaseSynapseType>;
    auto message_opt = calculate_lif_population_impl(population, endpoint, step_n);
    auto working_projections = find_projection_by_type_and_postsynaptic<StdpSynapseType, ProjectionContainer>(
        container, population.get_uid(), true);
    do_STDP_resource_plasticity(population, working_projections, message_opt, step_n);
    return message_opt;
}
}  // namespace knp::backends::cpu
